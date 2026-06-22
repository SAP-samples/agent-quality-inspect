import pytest
from pathlib import Path
from agent_inspect.metrics.adapters import (
    BaseAdapter,
    Tau2BenchAdapter,
    ToolsandboxAdapter,
)
from agent_inspect.models.metrics import AgentDialogueTrace, ToolInputParameter

# Tests for BaseAdapter


class TestBaseAdapter:
    def test_base_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()

    def test_base_adapter_convert_method_not_implemented(self):
        """Test that convert_to_agent_trace is abstract and must be implemented."""

        class IncompleteAdapter(BaseAdapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()


# Tests for Tau2BenchAdapter


class TestTau2BenchAdapter:
    @pytest.fixture
    def adapter(self):
        return Tau2BenchAdapter()

    @pytest.fixture
    def sample_data(self, adapter):
        sample_file = Path(__file__).parent / "sample_data" / "single_trace_tau2bench.json"
        return adapter.load_json(sample_file)

    def test_adapter_instantiation(self, adapter):
        assert isinstance(adapter, Tau2BenchAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_load_json(self, sample_data):
        assert isinstance(sample_data, list)
        assert len(sample_data) > 0

    def test_convert_to_agent_trace(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)

        # Basic strcture validation
        assert isinstance(converted_trace, AgentDialogueTrace)
        assert len(converted_trace.turns) == len(sample_data)

    def test_turn_0(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)
        turn_0 = converted_trace.turns[0]

        # Validate first turn structure
        assert len(turn_0.steps) == 0  # No tool calls in the 0th turn

        assert (
            turn_0.agent_input
            == "Hi, I\u2019d like to book a one-way flight from Chicago O\u2019Hare (ORD) to Philadelphia (PHL) on May 26. I want the exact same flight I took on May 10. My user id is sophia_silva_7557. I\u2019d also like to add an extra passenger, Kevin Smith (DOB: 2001-04-12). We prefer economy seats, specifically an aisle and a middle seat together. I don\u2019t need any checked bags or travel insurance. I\u2019ll pay with one of my certificates, and my budget is up to $500."
        )
        assert (
            turn_0.agent_response.response
            == "Here’s a summary of your booking request:\n\n- User ID: sophia_silva_7557\n- Trip type: One-way\n- Origin: ORD (Chicago O’Hare)\n- Destination: PHL (Philadelphia)\n- Date: May 26\n- Cabin class: Economy\n- Passengers:\n  - Sophia Silva (your info from profile)\n  - Kevin Smith (DOB: 2001-04-12)\n- Seats: Prefer aisle and middle together\n- Checked bags: None\n- Travel insurance: No\n- Payment: Certificate (in your profile)\n- Budget: Up to $500\n\nTo proceed, I’ll:\n1. Find the flight you took on May 10 from ORD to PHL.\n2. Search for the same flight on May 26 in economy class.\n\nCould you please confirm if all these details are correct and that Sophia Silva’s date of birth, as listed in your profile, is also correct? If confirmed, I’ll proceed to the next step."
        )

    def test_turn_1(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)
        turn_1 = converted_trace.turns[1]

        for i in range(len(turn_1.steps)):
            print(f"Step {i}:")
            print(turn_1.steps[i])
        # Validate first turn structure
        assert len(turn_1.steps) == 7  # 6 tool calls + 1 agent response

        # 0th step check
        assert turn_1.steps[0].tool == "get_user_details"
        assert turn_1.steps[0].tool_input_args[0] == ToolInputParameter(
            name="user_id", value="sophia_silva_7557", check=None
        )
        assert (
            turn_1.steps[0].tool_output
            == '{"user_id": "sophia_silva_7557", "name": {"first_name": "Sophia", "last_name": "Silva"}, "address": {"address1": "141 Cedar Avenue", "address2": "Suite 436", "city": "Columbus", "country": "USA", "state": "OH", "zip": "43282"}, "email": "sophia.silva5929@example.com", "dob": "1957-10-05", "payment_methods": {"certificate_8045380": {"source": "certificate", "id": "certificate_8045380", "amount": 500.0}, "certificate_3887113": {"source": "certificate", "id": "certificate_3887113", "amount": 150.0}, "credit_card_4196779": {"source": "credit_card", "id": "credit_card_4196779", "brand": "visa", "last_four": "7238"}, "gift_card_5094406": {"source": "gift_card", "id": "gift_card_5094406", "amount": 274.0}}, "saved_passengers": [{"first_name": "Yara", "last_name": "Silva", "dob": "1972-06-16"}, {"first_name": "Isabella", "last_name": "Moore", "dob": "1988-06-04"}], "membership": "regular", "reservations": ["NM1VX1", "KC18K6", "S61CZX", "H8Q05L", "WUNA5K"]}'
        )

        # 1st step check
        assert turn_1.steps[1].tool == "get_reservation_details"
        assert turn_1.steps[1].tool_input_args[0] == ToolInputParameter(
            name="reservation_id", value="NM1VX1", check=None
        )
        assert (
            turn_1.steps[1].tool_output
            == '{"reservation_id": "NM1VX1", "user_id": "sophia_silva_7557", "origin": "MSP", "destination": "EWR", "flight_type": "round_trip", "cabin": "basic_economy", "flights": [{"flight_number": "HAT300", "origin": "MSP", "destination": "EWR", "date": "2024-05-25", "price": 100}, {"flight_number": "HAT208", "origin": "EWR", "destination": "MSP", "date": "2024-05-27", "price": 53}], "passengers": [{"first_name": "Yara", "last_name": "Silva", "dob": "1972-06-16"}], "payment_history": [{"payment_id": "gift_card_5094406", "amount": 183}], "created_at": "2024-05-03T08:46:43", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "yes", "status": null}'
        )

        # 2nd step check
        assert turn_1.steps[2].tool == "get_reservation_details"
        assert turn_1.steps[2].tool_input_args[0] == ToolInputParameter(
            name="reservation_id", value="KC18K6", check=None
        )
        assert (
            turn_1.steps[2].tool_output
            == '{"reservation_id": "KC18K6", "user_id": "sophia_silva_7557", "origin": "MSP", "destination": "CLT", "flight_type": "one_way", "cabin": "basic_economy", "flights": [{"flight_number": "HAT300", "origin": "MSP", "destination": "EWR", "date": "2024-05-21", "price": 55}, {"flight_number": "HAT215", "origin": "EWR", "destination": "CLT", "date": "2024-05-21", "price": 51}], "passengers": [{"first_name": "Sophia", "last_name": "Silva", "dob": "1957-10-05"}], "payment_history": [{"payment_id": "credit_card_4196779", "amount": 136}], "created_at": "2024-05-04T14:07:11", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "yes", "status": null}'
        )

        # 3rd step check
        assert turn_1.steps[3].tool == "get_reservation_details"
        assert turn_1.steps[3].tool_input_args[0] == ToolInputParameter(
            name="reservation_id", value="S61CZX", check=None
        )
        assert (
            turn_1.steps[3].tool_output
            == '{"reservation_id": "S61CZX", "user_id": "sophia_silva_7557", "origin": "LAX", "destination": "CLT", "flight_type": "round_trip", "cabin": "economy", "flights": [{"flight_number": "HAT228", "origin": "LAX", "destination": "EWR", "date": "2024-05-23", "price": 131}, {"flight_number": "HAT043", "origin": "EWR", "destination": "CLT", "date": "2024-05-24", "price": 163}, {"flight_number": "HAT157", "origin": "CLT", "destination": "EWR", "date": "2024-05-24", "price": 157}, {"flight_number": "HAT041", "origin": "EWR", "destination": "LAX", "date": "2024-05-25", "price": 186}], "passengers": [{"first_name": "Harper", "last_name": "Jackson", "dob": "1974-09-05"}], "payment_history": [{"payment_id": "credit_card_4196779", "amount": 667}], "created_at": "2024-05-02T04:38:01", "total_baggages": 1, "nonfree_baggages": 0, "insurance": "yes", "status": null}'
        )

        # 4th step check
        assert turn_1.steps[4].tool == "get_reservation_details"
        assert turn_1.steps[4].tool_input_args[0] == ToolInputParameter(
            name="reservation_id", value="H8Q05L", check=None
        )
        assert (
            turn_1.steps[4].tool_output
            == '{"reservation_id": "H8Q05L", "user_id": "sophia_silva_7557", "origin": "JFK", "destination": "ATL", "flight_type": "one_way", "cabin": "basic_economy", "flights": [{"flight_number": "HAT268", "origin": "JFK", "destination": "ATL", "date": "2024-05-24", "price": 74}], "passengers": [{"first_name": "Harper", "last_name": "Kovacs", "dob": "1973-10-26"}], "payment_history": [{"payment_id": "credit_card_4196779", "amount": 104}], "created_at": "2024-05-03T15:12:00", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "yes", "status": null}'
        )

        # 5th step check
        assert turn_1.steps[5].tool == "get_reservation_details"
        assert turn_1.steps[5].tool_input_args[0] == ToolInputParameter(
            name="reservation_id", value="WUNA5K", check=None
        )
        assert (
            turn_1.steps[5].tool_output
            == '{"reservation_id": "WUNA5K", "user_id": "sophia_silva_7557", "origin": "ORD", "destination": "PHL", "flight_type": "round_trip", "cabin": "economy", "flights": [{"flight_number": "HAT271", "origin": "ORD", "destination": "PHL", "date": "2024-05-10", "price": 160}, {"flight_number": "HAT197", "origin": "PHL", "destination": "ORD", "date": "2024-05-11", "price": 100}], "passengers": [{"first_name": "Sophia", "last_name": "Silva", "dob": "1957-10-05"}], "payment_history": [{"payment_id": "gift_card_5094406", "amount": 260}], "created_at": "2024-05-08T19:01:02", "total_baggages": 1, "nonfree_baggages": 0, "insurance": "no", "status": null}'
        )

        # 6th step check
        assert turn_1.steps[6].tool == "search_direct_flight"
        assert turn_1.steps[6].tool_input_args[0] == ToolInputParameter(
            name="origin", value="ORD", check=None
        )
        assert turn_1.steps[6].tool_input_args[1] == ToolInputParameter(
            name="destination", value="PHL", check=None
        )
        assert turn_1.steps[6].tool_input_args[2] == ToolInputParameter(
            name="date", value="2024-05-26", check=None
        )
        assert (
            turn_1.steps[6].tool_output
            == '[{"flight_number": "HAT139", "origin": "ORD", "destination": "PHL", "status": "available", "scheduled_departure_time_est": "17:00:00", "scheduled_arrival_time_est": "19:00:00", "date": null, "available_seats": {"basic_economy": 7, "economy": 1, "business": 7}, "prices": {"basic_economy": 65, "economy": 114, "business": 395}}, {"flight_number": "HAT271", "origin": "ORD", "destination": "PHL", "status": "available", "scheduled_departure_time_est": "19:00:00", "scheduled_arrival_time_est": "21:00:00", "date": null, "available_seats": {"basic_economy": 20, "economy": 3, "business": 14}, "prices": {"basic_economy": 83, "economy": 174, "business": 338}}, {"flight_number": "HAT289", "origin": "ORD", "destination": "PHL", "status": "available", "scheduled_departure_time_est": "05:00:00", "scheduled_arrival_time_est": "07:00:00", "date": null, "available_seats": {"basic_economy": 17, "economy": 14, "business": 10}, "prices": {"basic_economy": 83, "economy": 161, "business": 253}}]'
        )


# Tests for ToolsandboxAdapter


class TestToolsandboxAdapter:
    @pytest.fixture
    def adapter(self):
        return ToolsandboxAdapter()

    @pytest.fixture
    def sample_data(self, adapter):
        sample_file = Path(__file__).parent / "sample_data" / "single_trace_toolsandbox.json"
        return adapter.load_json(sample_file)

    def test_adapter_instantiation(self, adapter):
        assert isinstance(adapter, ToolsandboxAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_load_json(self, sample_data):
        assert isinstance(sample_data, list)
        assert len(sample_data) > 0

    def test_convert_to_agent_trace(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)

        # Basic strcture validation
        assert isinstance(converted_trace, AgentDialogueTrace)
        assert len(converted_trace.turns) == len(sample_data)

    def test_turn_0(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)
        turn_0 = converted_trace.turns[0]

        # Validate first turn structure
        assert len(turn_0.steps) == 2  # 2 tool calls
        assert (
            turn_0.agent_input
            == "Please create a reminder for me to buy chocolate milk on March 22, 2024, at 5PM."
        )
        assert (
            turn_0.agent_response.response
            == "Your reminder to buy chocolate milk has been set for March 22, 2024, at 5:00 PM. Would you like to add any other details to this reminder?"
        )

        # Validate tool calls
        assert turn_0.steps[0].tool == "datetime_info_to_timestamp"
        assert turn_0.steps[0].tool_input_args[0] == ToolInputParameter(
            name="year", value=2024, check=None
        )
        assert turn_0.steps[0].tool_input_args[1] == ToolInputParameter(
            name="month", value=3, check=None
        )
        assert turn_0.steps[0].tool_input_args[2] == ToolInputParameter(
            name="day", value=22, check=None
        )
        assert turn_0.steps[0].tool_input_args[3] == ToolInputParameter(
            name="hour", value=17, check=None
        )
        assert turn_0.steps[0].tool_input_args[4] == ToolInputParameter(
            name="minute", value=0, check=None
        )
        assert turn_0.steps[0].tool_input_args[5] == ToolInputParameter(
            name="second", value=0, check=None
        )
        assert turn_0.steps[0].tool_output == "1711098000.0"

        assert turn_0.steps[1].tool == "add_reminder"
        assert turn_0.steps[1].tool_input_args[0] == ToolInputParameter(
            name="content", value="Buy chocolate milk", check=None
        )
        assert turn_0.steps[1].tool_input_args[1] == ToolInputParameter(
            name="reminder_timestamp", value=1711098000, check=None
        )

    def test_turn_1(self, adapter, sample_data):
        converted_trace = adapter.convert_to_agent_trace(sample_data)
        turn_1 = converted_trace.turns[1]

        # Validate second turn structure
        assert len(turn_1.steps) == 0  # no tool calls
        assert turn_1.agent_input == "No, that's all I needed. Thank you!"
        assert (
            turn_1.agent_response.response
            == "You\u2019re welcome! Your reminder is all set. If you need anything else, just let me know. Have a great day!"
        )
