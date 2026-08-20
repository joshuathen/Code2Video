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
        lecture_lines = [
            "We represent MLP transformation as a function.",
            "Input patterns match via the first weight.",
            "Fact injection occurs through the second weight.",
            "The signal flows through a bottleneck.",
            "Facts are distributed across these weights."
        ]
        self.setup_layout("Visualizing Fact Retrieval", lecture_lines)
        
        # Load asset
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        self.place_at_grid(box, "C3", scale_factor=1.0)
        
        # --- Animation for Lecture Line 1 ---
        self.play(FadeIn(box))
        self.lecture[0].set_color("#FFFFFF")
        box.set_color("#FFFFFF")
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        input_vector = Dot(color="#FF0000").next_to(box, LEFT)
        self.play(FadeIn(input_vector), box.animate.set_color("#FF0000"))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)

        # --- Animation for Lecture Line 4 ---
        output_projection = Text("Output", color="#00FFFF", font_size=24)
        self.place_at_grid(output_projection, "D6", scale_factor=0.7)
        self.play(Write(output_projection), box.animate.set_color("#00FFFF"))
        self.lecture[3].set_color("#00FFFF")
        self.wait(1)

        # --- Animation for Lecture Line 5 ---
        self.play(FadeOut(box), FadeOut(input_vector), FadeOut(output_projection))
        self.lecture[4].set_color("#FF00FF")
        self.wait(1)
