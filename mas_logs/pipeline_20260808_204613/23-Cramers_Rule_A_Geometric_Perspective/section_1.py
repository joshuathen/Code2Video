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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: Determinants as Area", [
            "Determinants measure the area of parallelograms.",
            "Columns of A define a base parallelogram.",
            "Determinants represent this total signed area."
        ])
        
        # Axes for the parallelogram
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 4, 1], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'B2', 'E4', scale_factor=0.6)
        
        u = np.array([2, 0, 0])
        v = np.array([0, 2, 0])
        
        vec_u = Vector(u, color="#FF5733")
        vec_v = Vector(v, color="#33FF57")
        
        # Parallelogram using Polygon
        points = [ORIGIN, u, u + v, v]
        para = Polygon(*points, color=BLUE, fill_opacity=0.3)
        para.shift(axes.c2p(0, 0) - ORIGIN)
        
        # Ensure vectors are aligned with axes
        vec_u.put_start_and_end_on(axes.c2p(0, 0), axes.c2p(*u[:2]))
        vec_v.put_start_and_end_on(axes.c2p(0, 0), axes.c2p(*v[:2]))
        
        # Label Assets
        # Note: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg exists
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        label_group = VGroup(icon, Text("Scaling Factor", font_size=18)).arrange(DOWN)
        self.place_in_area(label_group, 'A5', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF99")
        self.play(Create(axes), Create(vec_u), Create(vec_v), FadeIn(para))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF99")

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF99")
        formula = MathTex(r"|\det(A)| = \text{area} = 4").scale(0.8).set_color(WHITE)
        self.place_at_grid(formula, 'E3', scale_factor=0.9)
        self.play(Write(formula))
        self.play(FadeIn(label_group))
        self.play(Indicate(para, color="#FFFF00"), Indicate(formula, color="#FFFF00"))
        self.wait(2)
