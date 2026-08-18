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
        lecture_lines = ["Neural networks map inputs to outputs.", "Learning adjusts parameters to minimize errors.", "Think of a robot throwing a ball."]
        self.setup_layout("Prerequisite: The Mental Model (0:30)", lecture_lines)
        
        # Define elements
        box_input = SurroundingRectangle(Text("Input", font_size=24), buff=0.2, color=WHITE)
        box_proc = SurroundingRectangle(Text("Processing", font_size=24), buff=0.2, color=WHITE)
        box_output = SurroundingRectangle(Text("Output", font_size=24), buff=0.2, color=WHITE)
        
        # Add assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Improve layout per instructions
        self.place_at_grid(box_input, 'B2', scale_factor=0.9)
        self.place_at_grid(box_proc, 'C2', scale_factor=0.9)
        self.place_at_grid(box_output, 'D2', scale_factor=0.9)
        
        self.place_at_grid(robot, 'C3', scale_factor=0.5)
        
        group_boxes = VGroup(box_input, box_proc, box_output)
        self.place_in_area(group_boxes, 'B2', 'D4', scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"), Create(box_input), Create(box_proc), Create(box_output), FadeIn(robot))
        
        # === Animation for Lecture Line 2 ===
        label_cost = Text("Cost", font_size=20, color=RED)
        self.place_at_grid(label_cost, 'C3', scale_factor=0.7)
        
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"), FadeIn(label_cost))
        self.play(box_proc.animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(ball, 'C2', scale_factor=0.3)
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"), FadeIn(ball))
        self.play(ball.animate.shift(UP*0.2), ball.animate.shift(DOWN*0.2), run_time=1)
        self.wait(1)
