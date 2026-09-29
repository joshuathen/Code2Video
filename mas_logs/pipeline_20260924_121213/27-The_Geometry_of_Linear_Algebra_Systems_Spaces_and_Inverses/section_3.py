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
        lecture_lines = ["Null space contains vectors where Ax=0.", "These directions collapse space onto the origin.", "Analogy: Flattening 3D objects onto 2D."]
        self.setup_layout("Null Space: The 'Ghost' Vectors", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # 3D vector 'x'
        x_vec = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*0.5 + OUT*0.5, color="#FFA500")
        x_label = MathTex("x").set_color("#FFA500").next_to(x_vec, RIGHT)
        group1 = VGroup(x_vec, x_label)
        self.place_at_grid(group1, 'B4', scale_factor=0.6)
        self.play(FadeIn(group1))
        self.lecture[0].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show collapsing vectors
        collapsing_vecs = VGroup(*[Arrow(start=ORIGIN, end=RIGHT*1.2 + UP*0.4, color="#8B0000") for _ in range(3)])
        collapsing_vecs.arrange(DOWN, buff=0.1)
        self.place_at_grid(collapsing_vecs, 'D4', scale_factor=0.5)
        origin_dot = Dot(color=WHITE)
        self.place_at_grid(origin_dot, 'D5', scale_factor=1.0)
        self.play(Create(collapsing_vecs), FadeIn(origin_dot))
        self.play(collapsing_vecs.animate.move_to(origin_dot.get_center()))
        self.lecture[1].set_color("#8B0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flattening box
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        box.set_fill("#778899", opacity=1)
        label = Text("Flattened Object", font_size=18, color="#778899")
        self.place_at_grid(box, 'E4', scale_factor=0.5)
        self.place_at_grid(label, 'F4', scale_factor=0.5)
        
        self.play(FadeIn(box), FadeIn(label))
        self.play(box.animate.scale([1, 0.1, 1]))
        self.lecture[2].set_color("#778899")
        self.wait(2)
