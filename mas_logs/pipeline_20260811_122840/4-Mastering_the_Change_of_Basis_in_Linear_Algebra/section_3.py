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
        self.setup_layout("The Mechanism: How to Calculate", 
                          ["Step 1: Identify your basis vectors.", 
                           "Step 2: Place them as columns in P.", 
                           "Step 3: Multiply P by coordinate vector."])
        
        # Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arm.svg")
        check = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/item.svg")
        
        # Color objects
        robot.set_color(BLUE)
        arm.set_color(GREEN)
        check.set_color(YELLOW)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        steps = VGroup(
            Text("1. Identify B and B'", font_size=20),
            Text("2. Form P", font_size=20),
            Text("3. Calculate", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(steps, 'B5', scale_factor=0.8)
        self.place_at_grid(robot, 'A5', scale_factor=0.7)
        self.play(FadeIn(steps), FadeIn(robot))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        matrix_p = MathTex("P = [v_1 | v_2]").set_color(GREEN)
        self.place_at_grid(matrix_p, 'D5', scale_factor=0.8)
        self.play(Write(matrix_p))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        mult = MathTex("P \\cdot v_{robot} = v_{world}").set_color(YELLOW)
        self.place_at_grid(mult, 'E5', scale_factor=0.7)
        self.place_at_grid(arm, 'F2', scale_factor=0.7)
        self.place_at_grid(check, 'F6', scale_factor=0.7)
        self.play(Write(mult), FadeIn(arm), FadeIn(check))
        self.wait(2)
