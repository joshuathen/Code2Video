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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup lines
        lines = [
            'Nearby phones exchange Ephemeral IDs using Bluetooth.',
            "Bob’s phone stores Alice's ID and signal strength locally.",
            'This record stays on the device for fourteen days.'
        ]
        self.setup_layout("Step 2: The Bluetooth Handshake", lines)

        # Assets
        phone_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"
        calendar_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/calendar.svg"

        # Colors
        BLUE_WAVE = "#3498DB"
        GREEN_BAR = "#2ECC71"
        HIGHLIGHT_YELLOW = "#F1C40F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_WAVE)
        
        alice_phone = SVGMobject(phone_path, color=WHITE)
        bob_phone = SVGMobject(phone_path, color=WHITE)
        self.place_at_grid(alice_phone, "B2", scale_factor=0.6)
        self.place_at_grid(bob_phone, "B5", scale_factor=0.6)
        
        alice_label = Text("Alice", font_size=18).next_to(alice_phone, DOWN, buff=0.1)
        bob_label = Text("Bob", font_size=18).next_to(bob_phone, DOWN, buff=0.1)
        
        # Bluetooth Wave
        wave = Arc(radius=0.5, start_angle=-PI/3, angle=2*PI/3, color=BLUE_WAVE, stroke_width=4)
        wave.move_to(self.grid["B3"])
        
        ephid_a = Text("EphID_A", font_size=20, color=BLUE_WAVE)
        self.place_at_grid(ephid_a, "B2")

        self.play(FadeIn(alice_phone), FadeIn(bob_phone), FadeIn(alice_label), FadeIn(bob_label))
        
        # Pulse animation
        self.play(
            Create(wave),
            wave.animate.scale(1.5).set_stroke(opacity=0),
            run_time=1.0,
            rate_func=linear
        )
        self.remove(wave)
        
        # Move EphID from Alice to Bob
        self.play(ephid_a.animate.move_to(self.grid["B5"]), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN_BAR)
        
        # Local Log Table
        table_title = Text("Bob's Local Log", font_size=20)
        self.place_at_grid(table_title, "D3", scale_factor=0.8)
        
        # Table UI
        table_rect = Rectangle(width=4, height=1.5, color=WHITE)
        self.place_in_area(table_rect, "E2", "F5")
        
        header_id = Text("ID", font_size=16).move_to(table_rect.get_top() + DOWN * 0.3 + LEFT * 1.0)
        header_strength = Text("Strength", font_size=16).move_to(table_rect.get_top() + DOWN * 0.3 + RIGHT * 1.0)
        line_sep = Line(table_rect.get_left(), table_rect.get_right(), color=WHITE).shift(UP * 0.2)
        line_sep.move_to(table_rect.get_top() + DOWN * 0.5)
        
        log_group = VGroup(table_rect, table_title, header_id, header_strength, line_sep)
        
        # Entry data
        entry_id = Text("EphID_A", font_size=16, color=WHITE)
        entry_id.move_to(header_id.get_bottom() + DOWN * 0.4)
        
        strength_bar = Rectangle(width=1.2, height=0.2, color=GREEN_BAR, fill_opacity=0.8)
        strength_bar.move_to(header_strength.get_bottom() + DOWN * 0.4)
        
        entry_group = VGroup(entry_id, strength_bar)
        
        self.play(FadeIn(log_group))
        self.play(Transform(ephid_a, entry_id)) # EphID_A moves into table
        self.play(FadeIn(strength_bar))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_YELLOW)
        
        calendar = SVGMobject(calendar_path, color=HIGHLIGHT_YELLOW)
        self.place_at_grid(calendar, "D5", scale_factor=0.5)
        
        days_left = Integer(14, font_size=24, color=HIGHLIGHT_YELLOW, mob_class=Text).next_to(calendar, RIGHT, buff=0.2)
        days_label = Text("Days", font_size=18, color=HIGHLIGHT_YELLOW).next_to(days_left, DOWN, buff=0.1)
        
        calendar_group = VGroup(calendar, days_left, days_label)
        
        self.play(FadeIn(calendar_group))
        
        # Countdown
        for i in range(13, -1, -1):
            self.play(days_left.animate.set_value(i), run_time=0.1)
        
        self.wait(0.5)
        
        # At day 0, the log entry fades out
        self.play(
            FadeOut(ephid_a),
            FadeOut(strength_bar),
            entry_id.animate.set_color(RED), # Visual cue for deletion
            run_time=1.0
        )
        self.play(FadeOut(entry_id))
        
        self.wait(2)
