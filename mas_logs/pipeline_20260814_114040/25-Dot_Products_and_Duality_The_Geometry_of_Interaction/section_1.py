from manim import *
import numpy as np

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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are arrows representing direction and magnitude.",
            "Dot product projects one vector onto another.",
            "The shadow length measures the projection magnitude.",
            "This reduces 2D interaction to 1D scalar.",
            "The dot product captures their geometric alignment."
        ]
        self.setup_layout("Prerequisite: The Projection Intuition", lecture_lines)
        
        # Define objects
        vec_v = Vector([1.5, 1.0], color="#FF5733")
        vec_w = Vector([2.0, 0], color=WHITE)
        
        # Load Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.place_at_grid(vec_v, "B4", scale_factor=0.7)
        self.play(Create(vec_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.place_at_grid(vec_w, "C4", scale_factor=0.7)
        self.play(Create(vec_w))
        
        # Ruler asset for projection line
        self.place_at_grid(ruler, "D4", scale_factor=0.5)
        proj_line = DashedLine(vec_v.get_end(), [vec_v.get_end()[0], vec_w.get_start()[1], 0], color="#33FF57")
        self.add(proj_line, ruler)
        self.play(Create(proj_line), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        proj_point = Dot(point=[vec_v.get_end()[0], vec_w.get_start()[1], 0], color="#3357FF")
        self.play(FadeIn(proj_point))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.place_at_grid(flashlight, "E5", scale_factor=0.5)
        self.play(FadeIn(flashlight))
        self.wait(1)
        
        # Final fade out
        self.play(FadeOut(VGroup(vec_v, vec_w, proj_line, proj_point, ruler, flashlight)))
