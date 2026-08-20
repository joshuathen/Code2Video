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
        lecture_lines = ["We seek roots where a curve crosses zero.", "Start with a guess on the curve.", "Draw a tangent line at that point.", "Follow the tangent down to the axis.", "Repeat this to find the root."]
        self.setup_layout("The Quest for the Root (Prerequisites & Intuition)", lecture_lines)
        
        axes = Axes(x_range=[-2, 5], y_range=[-2, 5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * (x - 2)**2 - 1, color=WHITE)
        root_dot = Dot(axes.c2p(2, -1), color="#FF0000")
        
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(root_dot))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        x0 = 4
        p0 = Dot(axes.c2p(x0, 0.5 * (x0 - 2)**2 - 1), color="#00FFFF")
        pencil_obj = self.place_at_grid(pencil, "B3", scale_factor=0.3)
        pencil_obj.next_to(p0, UP)
        self.play(FadeIn(p0), FadeIn(pencil_obj))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # tangent line approximation
        tangent = Line(axes.c2p(3, -0.5), axes.c2p(5, 1.5), color="#FFFF00")
        ruler_obj = self.place_at_grid(ruler, "D4", scale_factor=0.3)
        self.play(Create(tangent), FadeIn(ruler_obj))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        x1 = 2
        p1 = Dot(axes.c2p(x1, 0), color="#00FF00")
        self.play(MoveAlongPath(p1, tangent), run_time=2)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.play(FadeOut(pencil_obj), FadeOut(ruler_obj))
