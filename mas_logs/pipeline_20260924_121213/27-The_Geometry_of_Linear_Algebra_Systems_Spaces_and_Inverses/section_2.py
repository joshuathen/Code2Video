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
        self.setup_layout("Column Space: Where Can We Reach?", [
            "Column space defines the reachable space.",
            "It is the span of column vectors.",
            "Robot arm segments represent column vectors."
        ])

        # Define vectors
        c1 = Vector([1, 2], color="#FFD700")
        c2 = Vector([2, 0.5], color="#7DF9FF")
        l1 = MathTex("c_1", color="#FFD700").next_to(c1.get_end(), UP, buff=0.1)
        l2 = MathTex("c_2", color="#7DF9FF").next_to(c2.get_end(), DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(c1, l1), FadeIn(c2, l2))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#7DF9FF")
        parallelogram = Polygon(
            ORIGIN, c1.get_end(), c1.get_end() + c2.get_end(), c2.get_end(),
            fill_opacity=0.3, fill_color="#D3D3D3", stroke_width=0
        )
        label = Text("Column Space", font_size=20, color="#D3D3D3").move_to(self.grid['D3'])
        self.play(Create(parallelogram), Write(label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'D5', scale_factor=0.3)
        self.add(robot)
        
        # Animate robot moving within the reach
        self.play(robot.animate.move_to(c1.get_end() * 0.5 + c2.get_end() * 0.5), run_time=2)
        self.play(robot.animate.move_to(c1.get_end() * 0.8 + c2.get_end() * 0.2), run_time=2)
        self.wait(1)
