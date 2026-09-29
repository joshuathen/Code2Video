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
        self.setup_layout("Visualizing e^(iπ)", ["Set x to pi radians.", "The robotic arm rotates halfway.", "We reach the negative real axis."])
        
        # Setup objects
        formula = MathTex(r"e^{i\theta} = \cos\theta + i \sin\theta").set_color(WHITE)
        self.place_at_grid(formula, 'B5', scale_factor=0.9)
        
        # Load asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, "B3", scale_factor=0.4)
        
        circle = Circle(radius=1.5, color=BLUE)
        self.place_at_grid(circle, 'D5', scale_factor=0.7)
        
        arm = Line(start=ORIGIN, end=RIGHT*1.5, color=YELLOW)
        arm.move_to(self.grid["D5"])
        
        dot = Dot(color=YELLOW)
        dot.move_to(arm.get_end())
        
        self.add(formula, robot, circle, arm, dot)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        formula_pi = MathTex(r"e^{i\pi} = \cos\pi + i \sin\pi").set_color("#FF0000")
        self.play(Transform(formula, formula_pi))
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # Animate arm rotation
        self.play(
            Rotate(arm, angle=PI, about_point=self.grid["D5"]), 
            MoveAlongPath(dot, Arc(radius=1.5, start_angle=0, angle=PI, arc_center=self.grid["D5"]))
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Add robot asset at the final point
        robot_final = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_final, "C1", scale_factor=0.4)
        self.play(FadeIn(robot_final))
        
        result_dot = Dot(color=RED, radius=0.15)
        result_dot.move_to(arm.get_end())
        self.play(Create(result_dot))
        
        final_text = MathTex(r"e^{i\pi} = -1").set_color("#00FF00")
        self.place_at_grid(final_text, 'B6', scale_factor=1.0)
        self.play(Write(final_text))
        
        self.wait(2)
