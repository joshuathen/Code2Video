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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Meaning of i and Rotation", [
            "Multiplying by 'i' rotates numbers ninety degrees.",
            "Repeated multiplication by 'i' creates rotation.",
            "This movement forms a perfect circle."
        ])
        
        # Axes
        axes = Axes(x_length=4, y_length=4, x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, "B2", "E5", scale_factor=0.8)
        self.add(axes)
        
        point = Dot(axes.c2p(1, 0), color=WHITE)
        label = MathTex("1").next_to(point, RIGHT)
        self.add(point, label)
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, "B2", scale_factor=0.3)
        self.add(compass)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        rotated_point = Dot(axes.c2p(0, 1), color="#00FFFF")
        rotated_label = MathTex("i", color="#00FFFF").next_to(rotated_point, UP)
        self.play(Rotate(point, angle=PI/2, about_point=axes.c2p(0, 0)), Transform(label, rotated_label))
        self.play(FadeIn(rotated_point))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Show movement to -1
        point_neg1 = Dot(axes.c2p(-1, 0), color="#FF00FF")
        label_neg1 = MathTex("-1", color="#FF00FF").next_to(point_neg1, LEFT)
        self.play(Rotate(point, angle=PI/2, about_point=axes.c2p(0, 0)), FadeIn(point_neg1), FadeIn(label_neg1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        circle = Circle(radius=axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0], color="#FF00FF", stroke_width=2).move_to(axes.c2p(0, 0))
        self.play(Create(circle))
        self.wait(2)
