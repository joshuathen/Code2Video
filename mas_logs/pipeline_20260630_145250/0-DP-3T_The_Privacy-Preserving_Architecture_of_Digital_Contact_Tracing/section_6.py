from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with Section 6 Title and Lecture Lines
        self.setup_layout(
            "Local Risk Assessment: The Matching Process",
            [
                "Phones periodically download the list of diagnosed keys.",
                "Each phone locally re-generates IDs from these keys.",
                "The phone compares these to its own encounter log.",
                "Matches indicate potential exposure to the virus.",
                "All calculations and notifications occur privately on-device."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Colors
        ALICE_COLOR = "#58D68D"
        KEY_COLOR = "#F39C12"
        RPI_COLOR = "#1ABC9C"
        LOG_COLOR = "#BDC3C7"
        MATCH_COLOR = "#F1C40F"
        ALERT_COLOR = "#E67E22"

        # Alice's Phone
        # Resolution for Issue 47: Shifted right to avoid crowding lecture notes
        alice_phone = RoundedRectangle(height=3.5, width=2.4, corner_radius=0.2, color=ALICE_COLOR)
        phone_label = Text("Alice's Phone", font_size=18, color=ALICE_COLOR)
        self.place_in_area(alice_phone, 'C3', 'E5')
        self.place_at_grid(phone_label, 'F4', scale_factor=0.8)
        
        # Bulletin Board (Source of Keys)
        # Resolution for Issue 46: Corrected overlap by placing label above board
        board = Rectangle(height=0.8, width=3.0, color=WHITE, fill_opacity=0.1)
        board_label = Text("Bulletin Board", font_size=16, color=WHITE)
        self.place_at_grid(board, 'B4')
        self.place_at_grid(board_label, 'A4', scale_factor=0.8)

        # Keys Downloading
        key_1 = Square(side_length=0.2, fill_opacity=1, color=KEY_COLOR)
        key_2 = Square(side_length=0.2, fill_opacity=1, color=KEY_COLOR)
        key_3 = Square(side_length=0.2, fill_opacity=1, color=KEY_COLOR)
        keys = VGroup(key_1, key_2, key_3).arrange(RIGHT, buff=0.2)
        self.place_at_grid(keys, 'B4')

        self.lecture[0].set_color(KEY_COLOR)
        self.play(Create(alice_phone), Create(phone_label), Create(board), Create(board_label))
        # Move keys to center of phone (D4)
        self.play(keys.animate.move_to(self.grid['D4']))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Alice's phone generates RPIs (#1ABC9C) from the keys.
        rpi_1 = Circle(radius=0.1, fill_opacity=1, color=RPI_COLOR)
        rpi_2 = Circle(radius=0.1, fill_opacity=1, color=RPI_COLOR)
        rpi_3 = Circle(radius=0.1, fill_opacity=1, color=RPI_COLOR)
        rpis = VGroup(rpi_1, rpi_2, rpi_3).arrange(RIGHT, buff=0.2)
        # Resolution for Issue 48: Placement in center column (D4)
        self.place_at_grid(rpis, 'D4')

        self.lecture[1].set_color(RPI_COLOR)
        self.play(Transform(keys, rpis))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Alice's 'Local Log' (#BDC3C7) is compared with generated RPIs.
        local_log_box = Rectangle(height=1.5, width=1.2, color=LOG_COLOR)
        log_title = Text("Local Log", font_size=14, color=LOG_COLOR)
        log_entry_1 = Line(LEFT*0.3, RIGHT*0.3, color=LOG_COLOR).shift(UP*0.3)
        log_entry_2 = Line(LEFT*0.3, RIGHT*0.3, color=LOG_COLOR)
        log_entry_3 = Line(LEFT*0.3, RIGHT*0.3, color=LOG_COLOR).shift(DOWN*0.3)
        local_log = VGroup(local_log_box, log_title.next_to(local_log_box, UP, buff=0.1), log_entry_1, log_entry_2, log_entry_3)
        
        # Position log inside phone (left column D3)
        # Resolution for Issue 47: Moved to D3
        self.place_at_grid(local_log, 'D3', scale_factor=0.7)
        # Move RPIs to the right side inside phone (D5) to avoid overlapping with log
        self.play(keys.animate.move_to(self.grid['D5']))
        
        self.lecture[2].set_color(LOG_COLOR)
        self.play(FadeIn(local_log))
        
        # Comparison indicator: Spans columns 3 to 5 (phone width)
        scan_line = Line(self.grid['D3'], self.grid['D5'], color=WHITE).set_stroke(width=2)
        self.play(scan_line.animate.shift(DOWN*0.5), run_time=1)
        self.play(scan_line.animate.shift(UP*0.5), run_time=1)
        self.play(FadeOut(scan_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A match is highlighted with a gold rectangle (#F1C40F).
        match_rect = SurroundingRectangle(log_entry_2, color=MATCH_COLOR, buff=0.1)
        match_text = Text("Match!", font_size=14, color=MATCH_COLOR)
        # Resolution for Issue 47: Moved to E3 (bottom-left of phone area)
        self.place_at_grid(match_text, 'E3', scale_factor=0.7)

        self.lecture[3].set_color(MATCH_COLOR)
        self.play(Create(match_rect), Write(match_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Risk Alert' (#E67E22) notification pops up on Alice's screen.
        alert_box = RoundedRectangle(height=0.6, width=1.8, corner_radius=0.1, color=ALERT_COLOR, fill_opacity=0.8)
        alert_text = Text("RISK ALERT", font_size=16, color=WHITE)
        alert_group = VGroup(alert_box, alert_text)
        # Resolution for Issue 48: Placed at C4 (top-center of phone) to avoid clutter
        self.place_at_grid(alert_group, 'C4', scale_factor=0.9)

        self.lecture[4].set_color(ALERT_COLOR)
        self.play(FadeIn(alert_group, shift=UP))
        self.play(Flash(alert_group, color=ALERT_COLOR))
        self.wait(2)
