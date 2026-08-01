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
        # Setup content
        lecture_lines = [
            "If Alice tests positive, she voluntarily shares her seed.",
            "She only uploads seeds for her contagious days.",
            "The central server receives these diagnosis keys anonymously.",
            "The server acts like a public bulletin board.",
            "It never learns who Alice actually met."
        ]
        self.setup_layout("Phase 2: The Diagnosis Key (1:00)", lecture_lines)

        # Assets Preparation
        # Alice Icon
        alice_circ = Circle(radius=0.4, color="#3498DB", fill_opacity=0.8)
        alice_label = Text("Alice", font_size=20).next_to(alice_circ, DOWN, buff=0.1)
        alice = VGroup(alice_circ, alice_label)
        # Issue 37: Increase scale to 1.3
        self.place_at_grid(alice, "C2", scale_factor=1.3)

        # Diagnosis Status (+)
        plus_sign = Text("+", color=WHITE, font_size=36).move_to(alice_circ.get_center())

        # Seed Icon
        seed_box = RoundedRectangle(height=0.4, width=0.8, color="#E67E22", fill_opacity=1)
        seed_text = Text("Seed", font_size=16, color=BLACK).move_to(seed_box.get_center())
        seed_group = VGroup(seed_box, seed_text)

        # Cloud Server Icon
        c1 = Circle(radius=0.4, color="#FFFFFF", fill_opacity=0.8).shift(LEFT*0.3)
        c2 = Circle(radius=0.5, color="#FFFFFF", fill_opacity=0.8).shift(UP*0.2)
        c3 = Circle(radius=0.4, color="#FFFFFF", fill_opacity=0.8).shift(RIGHT*0.3)
        cloud = VGroup(c1, c2, c3)
        cloud_label = Text("Central Server", font_size=18).next_to(cloud, DOWN, buff=0.2)
        cloud_full = VGroup(cloud, cloud_label)
        # Issue 38: Place in area C5-C6 with scale 1.3
        self.place_in_area(cloud_full, "C5", "C6", scale_factor=1.3)

        # Privacy labels
        no_id = Text("No Personal ID", font_size=14, color="#E74C3C")
        no_contacts = Text("No Contact List", font_size=14, color="#E74C3C")
        privacy_group = VGroup(no_id, no_contacts).arrange(DOWN, buff=0.1).next_to(cloud, UP, buff=0.2)

        # === Animation for Lecture Line 1 ===
        # If Alice tests positive, she voluntarily shares her seed.
        self.play(FadeIn(alice))
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.play(
            alice_circ.animate.set_color("#E74C3C"),
            FadeIn(plus_sign)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # She only uploads seeds for her contagious days.
        self.play(self.lecture[1].animate.set_color("#E67E22"))
        # Issue 39: scale factor 1.1
        self.place_at_grid(seed_group, "C2", scale_factor=1.1)
        self.play(FadeIn(seed_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The central server receives these diagnosis keys anonymously.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeIn(cloud_full))
        
        # Adjusting arrow start and end based on new scales and positions
        arrow_start = alice.get_right() + RIGHT*0.1
        arrow_end = cloud_full.get_left() + LEFT*0.1
        arrow = Arrow(start=arrow_start, end=arrow_end, color=WHITE, buff=0.1)
        
        self.play(GrowArrow(arrow))
        self.play(
            seed_group.animate.move_to(cloud.get_center()).scale(0.7),
            run_time=1.5
        )
        self.play(FadeOut(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The server acts like a public bulletin board.
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        bulletin_label = Text("Bulletin Board", font_size=18, color=YELLOW).next_to(cloud, DOWN, buff=0.2)
        self.play(Transform(cloud_label, bulletin_label))
        
        # Display seed pinned to the board
        self.play(seed_group.animate.shift(UP*0.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It never learns who Alice actually met.
        self.play(self.lecture[4].animate.set_color("#E74C3C"))
        self.play(FadeIn(privacy_group))
        
        # Cross them out to emphasize "never learns"
        cross1 = Line(no_id.get_left(), no_id.get_right(), color=RED, stroke_width=2)
        cross2 = Line(no_contacts.get_left(), no_contacts.get_right(), color=RED, stroke_width=2)
        self.play(Create(cross1), Create(cross2))
        
        self.wait(3)
