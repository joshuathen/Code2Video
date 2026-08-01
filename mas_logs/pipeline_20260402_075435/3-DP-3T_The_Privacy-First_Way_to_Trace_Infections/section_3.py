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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Phones broadcast temporary IDs via Bluetooth signals.",
            "Alice’s phone generates a new ID every fifteen minutes.",
            "Bob’s phone records Alice’s ID in a local log.",
            "No locations or names are ever exchanged.",
            "These secret handshakes stay only on your device."
        ]
        self.setup_layout("Phase 1: The Daily Handshake (1:00)", lecture_lines)
        
        # Colors
        ALICE_COLOR = "#3498DB"
        BOB_COLOR = "#E67E22"
        BT_RANGE_COLOR = "#5DADE2"
        CLOCK_COLOR = "#FFFFFF"
        ID_COLOR = "#F1C40F"
        LOG_COLOR = "#2ECC71"

        # === Animation for Lecture Line 1 ===
        # Phones broadcast temporary IDs via Bluetooth signals.
        self.lecture[0].set_color(ALICE_COLOR)
        
        alice_phone = RoundedRectangle(corner_radius=0.1, height=1.0, width=0.6, color=ALICE_COLOR)
        self.place_at_grid(alice_phone, "C2")
        alice_label = Text("Alice", font_size=18, color=ALICE_COLOR)
        self.place_at_grid(alice_label, "B2")
        
        bob_phone = RoundedRectangle(corner_radius=0.1, height=1.0, width=0.6, color=BOB_COLOR)
        self.place_at_grid(bob_phone, "C5")
        bob_label = Text("Bob", font_size=18, color=BOB_COLOR)
        self.place_at_grid(bob_label, "B5")
        
        bt_range = Circle(radius=1.8, color=BT_RANGE_COLOR, stroke_width=2).set_opacity(0.3)
        self.place_in_area(bt_range, "B2", "E5", scale_factor=1.0)
        
        self.play(
            FadeIn(alice_phone), FadeIn(alice_label),
            FadeIn(bob_phone), FadeIn(bob_label),
            Create(bt_range)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Alice’s phone generates a new ID every fifteen minutes.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CLOCK_COLOR)
        
        clock_circle = Circle(radius=0.3, color=CLOCK_COLOR)
        clock_hand = Line(ORIGIN, UP * 0.25, color=CLOCK_COLOR)
        clock = VGroup(clock_circle, clock_hand)
        self.place_at_grid(clock, "A4")
        
        id_a = Text("ID-A", font_size=20, color=ID_COLOR)
        self.place_at_grid(id_a, "C2", scale_factor=0.6) # Over Alice's phone, scaled to fit
        
        self.play(FadeIn(clock))
        self.play(Rotate(clock_hand, angle=-TAU, about_point=clock_circle.get_center()), run_time=1.5)
        self.play(Write(id_a))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Bob’s phone records Alice’s ID in a local log.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(LOG_COLOR)
        
        log_box = RoundedRectangle(corner_radius=0.05, height=1.2, width=0.8, color=LOG_COLOR)
        self.place_at_grid(log_box, "E5")
        log_title = Text("Log", font_size=14, color=LOG_COLOR)
        self.place_at_grid(log_title, "D5")
        
        # Move ID-A from Alice to Bob
        self.play(id_a.animate.move_to(self.grid["C5"]), run_time=1.5)
        
        log_entry_1 = Text("ID-A", font_size=14, color=ID_COLOR)
        log_entry_1.move_to(self.grid["E5"] + UP * 0.3)
        
        self.play(FadeIn(log_box), FadeIn(log_title))
        self.play(Transform(id_a, log_entry_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # No locations or names are ever exchanged.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        cross_line1 = Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(PI/4)
        cross_line2 = Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(-PI/4)
        cross = VGroup(cross_line1, cross_line2)
        
        loc_text = Text("GPS/Location", font_size=16, color=RED)
        name_text = Text("Identity/Name", font_size=16, color=RED)
        
        loc_group = VGroup(loc_text, cross.copy())
        name_group = VGroup(name_text, cross.copy())
        
        self.place_at_grid(loc_group, "B3", scale_factor=0.75)
        self.place_at_grid(name_group, "C3", scale_factor=0.75)
        
        self.play(FadeIn(loc_group), FadeIn(name_group))
        self.play(Indicate(loc_group), Indicate(name_group))
        self.wait(1)
        self.play(FadeOut(loc_group), FadeOut(name_group))

        # === Animation for Lecture Line 5 ===
        # These secret handshakes stay only on your device.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(LOG_COLOR)
        
        # Clock ticks again, send ID-B
        id_b = Text("ID-B", font_size=20, color=ID_COLOR)
        self.place_at_grid(id_b, "C2", scale_factor=0.6) # Over Alice's phone, scaled to fit
        
        log_entry_2 = Text("ID-B", font_size=14, color=ID_COLOR)
        log_entry_2.move_to(self.grid["E5"] + DOWN * 0.1)
        
        self.play(Rotate(clock_hand, angle=-TAU, about_point=clock_circle.get_center()), run_time=1)
        self.play(Write(id_b))
        self.play(id_b.animate.move_to(self.grid["C5"]), run_time=1.2)
        self.play(Transform(id_b, log_entry_2))
        
        # Emphasize storage is only on Bob's phone area
        self.play(Indicate(log_box), Indicate(bob_phone))
        self.wait(2)
