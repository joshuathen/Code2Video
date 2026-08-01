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

class Section6Scene(TeachingScene):
    def construct(self):
        title_text = "Summary and Real-World Impact"
        lecture_lines = [
            "Quantum states enable parallel processing in computers.",
            "Superposition allows exploring many paths at once.",
            "Quantum logic powers the next generation of technology."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for each stage
        color1 = BLUE_B
        color2 = GREEN_B
        color3 = "#FFD700" # Gold

        # Assets
        robot_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        # Recap showing vector |psi> and collapse to a single axis.
        self.play(self.lecture[0].animate.set_color(color1))
        
        axes = Axes(
            x_range=[0, 2, 1],
            y_range=[0, 2, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"color": WHITE, "include_tip": True}
        )
        # Position axes further from lecture notes (Issue 38)
        self.place_in_area(axes, "B2", "D4", scale_factor=0.8)
        
        psi_label = MathTex(r"|\psi\rangle", color=color1)
        # Repositioned psi_label (Issue 39)
        self.place_at_grid(psi_label, "B4", scale_factor=0.8)
        
        psi_vec = Arrow(axes.c2p(0,0), axes.c2p(1.2, 1.2), buff=0, color=color1)
        
        self.play(Create(axes), GrowArrow(psi_vec), Write(psi_label))
        self.wait(1)
        
        # Collapse to x-axis
        collapsed_vec = Arrow(axes.c2p(0,0), axes.c2p(1.2, 0), buff=0, color=color1)
        collapse_text = Text("Collapse", font_size=14, color=color1)
        # Repositioned collapse_text (Issue 39)
        self.place_at_grid(collapse_text, "D4", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(psi_vec, collapsed_vec),
            Write(collapse_text)
        )
        self.wait(1.5)
        
        # Clear recap elements for next stage
        self.play(FadeOut(axes), FadeOut(psi_label), FadeOut(collapsed_vec), FadeOut(collapse_text))

        # === Animation for Lecture Line 2 ===
        # Show a classical robot (one path) vs quantum robot (all paths).
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Maze paths representation
        paths = VGroup(
            Line(self.grid["B4"], self.grid["B6"], color=WHITE),
            Line(self.grid["C4"], self.grid["C6"], color=WHITE),
            Line(self.grid["D4"], self.grid["D6"], color=WHITE),
            Line(self.grid["E4"], self.grid["E6"], color=WHITE),
        )
        # Add entry/exit vertical lines for "maze" look
        entry_line = Line(self.grid["B4"], self.grid["E4"], color=WHITE)
        exit_line = Line(self.grid["B6"], self.grid["E6"], color=WHITE)
        
        maze = VGroup(paths, entry_line, exit_line)
        self.play(Create(maze))
        
        # Classical robot section using Asset (Issue 26)
        class_label = Text("Classical: Sequential", font_size=18, color=RED)
        self.place_at_grid(class_label, "A5", scale_factor=1.0)
        
        c_robot = SVGMobject(robot_svg_path).set_color(RED)
        self.place_at_grid(c_robot, "B4", scale_factor=0.3) # Scaled down for grid
        
        self.play(Write(class_label), FadeIn(c_robot))
        
        # Move through paths sequentially
        for i, row in enumerate(["B", "C", "D", "E"]):
            # Start of path
            if i > 0:
                self.play(c_robot.animate.move_to(self.grid[f"{row}4"]), run_time=0.2)
            # Traverse path
            self.play(c_robot.animate.move_to(self.grid[f"{row}6"]), run_time=0.4)
            
        self.play(FadeOut(c_robot), FadeOut(class_label))
        
        # Quantum robot section using Asset (Issue 26)
        quant_label = Text("Quantum: Parallel", font_size=18, color=color2)
        self.place_at_grid(quant_label, "A5", scale_factor=1.0)
        
        # Create base quantum robot once
        q_robot_base = SVGMobject(robot_svg_path).set_color(color2).scale(0.3)
        q_robots = VGroup(*[q_robot_base.copy() for _ in range(4)])
        
        for i, row in enumerate(["B", "C", "D", "E"]):
            q_robots[i].move_to(self.grid[f"{row}4"])
            
        self.play(Write(quant_label), FadeIn(q_robots))
        # All move simultaneously
        self.play(
            *[q_robots[i].animate.move_to(self.grid[f"{row}6"]) 
              for i, row in enumerate(["B", "C", "D", "E"])],
            run_time=1.5
        )
        self.wait(1.5)
        
        # Clear for final line
        self.play(FadeOut(maze), FadeOut(q_robots), FadeOut(quant_label))

        # === Animation for Lecture Line 3 ===
        # Fade in 'Parallel Processing Power' (gold #FFD700).
        self.play(self.lecture[2].animate.set_color(color3))
        
        final_text = Text("Parallel Processing Power", color=color3, font_size=32)
        # Position and scale adjusted (Issue 37)
        self.place_in_area(final_text, "C3", "E6", scale_factor=1.0)
        
        # Highlight effect
        box = SurroundingRectangle(final_text, color=color3, buff=0.3)
        
        self.play(Write(final_text), Create(box))
        self.play(Indicate(final_text, color=color3))
        self.wait(3)
