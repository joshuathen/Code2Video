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
        lecture_lines = ["Alice uploads her daily key.", "Bob's phone computes her IDs.", "Matches trigger an exposure alert."]
        self.setup_layout("The Diagnostic Path: Revealing Exposure", lecture_lines)
        
        # Elements
        alice_key = VGroup(Square(color=BLUE, fill_opacity=0.5), Text("Key", font_size=16)).arrange(DOWN)
        server = RoundedRectangle(color=WHITE, corner_radius=0.1, height=1, width=1.5).add(Text("Server", font_size=18))
        bob_phone = RoundedRectangle(color=GREEN, corner_radius=0.1, height=1, width=0.6).add(Text("Bob", font_size=14))
        match_icon = Star(color=RED, fill_opacity=0.8).scale(0.5)

        self.place_at_grid(alice_key, 'B2')
        self.place_at_grid(server, 'B4', scale_factor=0.8)
        self.place_at_grid(bob_phone, 'E4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(alice_key))
        self.play(alice_key.animate.move_to(self.grid['B4']))
        self.play(FadeOut(alice_key))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        bob_id_stream = VGroup(*[Circle(radius=0.1, color=GREEN, fill_opacity=0.5) for _ in range(5)]).arrange(RIGHT)
        self.place_at_grid(bob_id_stream, 'E3', scale_factor=0.7)
        self.play(Create(bob_id_stream))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_at_grid(match_icon, 'E5', scale_factor=0.9)
        self.play(FadeIn(match_icon), Flash(match_icon))
        self.play(Wait(2))
