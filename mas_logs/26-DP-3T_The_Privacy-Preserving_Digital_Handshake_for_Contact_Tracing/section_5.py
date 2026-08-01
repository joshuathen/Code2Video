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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "Phase 3: The Diagnosis & Upload"
        lecture_lines = [
            "A positive test result triggers a voluntary data upload.",
            "Alice uploads only her Secret Keys, not her contacts.",
            "The server adds these keys to a public list."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Elements creation
        alice_circle = Circle(radius=0.4, color="#FFD700", fill_opacity=0.8)
        alice_name = Text("Alice", font_size=20).next_to(alice_circle, UP, buff=0.1)
        alice = VGroup(alice_circle, alice_name)
        self.place_at_grid(alice, "B2")

        cloud_base = Circle(radius=0.5, color="#A9A9A9", fill_opacity=0.8)
        cloud_text = Text("Cloud", font_size=20, color=WHITE).move_to(cloud_base)
        cloud = VGroup(cloud_base, cloud_text)
        self.place_at_grid(cloud, "B5")

        self.add(alice, cloud)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # Change Alice's icon color to Red, add 'Positive' label, add 'Test Result' icon.
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        positive_label = Text("Positive", font_size=20, color="#FFFFFF")
        self.place_at_grid(positive_label, "C2")
        
        test_result = Rectangle(width=0.6, height=0.8, color="#FFFFFF", fill_opacity=0.2)
        test_text = Text("TEST", font_size=12, color="#FFFFFF").move_to(test_result)
        test_icon = VGroup(test_result, test_text)
        self.place_at_grid(test_icon, "B1")

        self.play(
            alice_circle.animate.set_color("#FF0000"),
            FadeIn(positive_label),
            FadeIn(test_icon)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Alice's phone sends a green 'Secret Key (SK)' (#00FF00) to a central Cloud icon (#A9A9A9).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )

        sk_box = RoundedRectangle(corner_radius=0.1, width=0.8, height=0.4, color="#00FF00", fill_opacity=1)
        sk_label = Text("SK", font_size=18, color=BLACK).move_to(sk_box)
        sk = VGroup(sk_box, sk_label)
        self.place_at_grid(sk, "B2")

        self.play(FadeIn(sk))
        self.play(sk.animate.move_to(self.grid["B5"]), run_time=1.5)
        self.play(FadeOut(sk))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Cloud icon displays a list titled 'Public Infected Keys' (#FF0000) and adds the SK to it.
        # Highlight the list on the Cloud, emphasizing no contact details or locations.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF0000")
        )

        list_box = Rectangle(width=2.5, height=2.0, color="#FF0000", fill_opacity=0.1)
        list_header = Text("Public Infected Keys", font_size=18, color="#FF0000")
        list_container = VGroup(list_header, list_box).arrange(DOWN, buff=0.1)
        self.place_in_area(list_container, "D4", "F6")

        key_entry_1 = Text("Key_72x...", font_size=16, color=WHITE)
        key_entry_2 = Text("Key_91a...", font_size=16, color=WHITE)
        key_entry_alice = Text("SK_Alice", font_size=16, color="#00FF00")
        list_entries = VGroup(key_entry_1, key_entry_2, key_entry_alice).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        list_entries.move_to(list_box.get_center())

        self.play(FadeIn(list_container))
        self.play(Write(key_entry_1), Write(key_entry_2))
        self.wait(0.5)
        
        # Adding Alice's key to the list
        self.place_at_grid(sk, "B5")
        sk.set_alpha(1)
        self.play(FadeIn(sk))
        self.play(sk.animate.move_to(key_entry_alice.get_center()).scale(0.5))
        self.play(FadeTransform(sk, key_entry_alice))

        # Highlight no contacts/locations
        no_info_box = Rectangle(width=3, height=1, color=YELLOW, fill_opacity=0.2)
        no_info_text = Text("No contacts or locations!", font_size=18, color=YELLOW)
        no_info = VGroup(no_info_box, no_info_text)
        self.place_at_grid(no_info, "E2")

        self.play(Create(no_info_box), Write(no_info_text))
        self.play(Indicate(list_container))
        
        self.wait(3)
