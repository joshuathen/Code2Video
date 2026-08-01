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
        # Initialize Scene
        lecture_lines = [
            "Qubits explore many paths at the exact same time.",
            "A quantum mouse searches every maze route simultaneously.",
            "This massive parallelism solves complex problems in one go."
        ]
        self.setup_layout("Why It Matters: Quantum Computing", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a maze outline (#FFFFFF) with a red dot trying paths one-by-one.
        
        maze_walls = VGroup()
        wall_color = "#FFFFFF"
        # Perimeter (Area A2 to F6)
        maze_walls.add(Line(self.grid["A2"] + LEFT*0.5 + UP*0.5, self.grid["A6"] + RIGHT*0.5 + UP*0.5, color=wall_color)) # Top
        maze_walls.add(Line(self.grid["F2"] + LEFT*0.5 + DOWN*0.5, self.grid["F6"] + RIGHT*0.5 + DOWN*0.5, color=wall_color)) # Bottom
        maze_walls.add(Line(self.grid["A2"] + LEFT*0.5 + UP*0.5, self.grid["F2"] + LEFT*0.5 + DOWN*0.5, color=wall_color)) # Left
        maze_walls.add(Line(self.grid["A6"] + RIGHT*0.5 + UP*0.5, self.grid["F6"] + RIGHT*0.5 + DOWN*0.5, color=wall_color)) # Right
        
        # Internal walls to define paths
        maze_walls.add(Line(self.grid["A3"] + RIGHT*0.5 + UP*0.5, self.grid["A3"] + RIGHT*0.5 + DOWN*0.5, color=wall_color)) # Block A3->A4
        maze_walls.add(Line(self.grid["C2"] + LEFT*0.5 + DOWN*0.5, self.grid["C2"] + RIGHT*0.5 + DOWN*0.5, color=wall_color)) # Block C2->D2

        exit_star = Star(n=5, color="#FFFF00", fill_opacity=1)
        # Issue 41 Fix: Positioned at F6 with scale 1.2
        self.place_at_grid(exit_star, "F6", scale_factor=1.2)
        
        # Highlight Lecture Line 1
        self.play(self.lecture[0].animate.set_color(RED))
        self.play(Create(maze_walls))
        self.add(exit_star)
        
        # Classical Mouse (Red Dot)
        # Issue 42 Fix: Starting at A2 with scale 1.2
        classical_mouse = Dot(color=RED)
        self.place_at_grid(classical_mouse, "A2", scale_factor=1.2)
        self.play(FadeIn(classical_mouse))
        
        # Sequence: path 1 -> dead end -> backtrack -> path 2 -> dead end
        self.play(classical_mouse.animate.move_to(self.grid["A3"]), run_time=0.6)
        self.play(Indicate(classical_mouse, color=RED))
        self.play(classical_mouse.animate.move_to(self.grid["A2"]), run_time=0.4)
        
        self.play(classical_mouse.animate.move_to(self.grid["B2"]), run_time=0.6)
        self.play(classical_mouse.animate.move_to(self.grid["C2"]), run_time=0.6)
        self.play(Indicate(classical_mouse, color=RED))
        
        self.play(FadeOut(classical_mouse))

        # === Animation for Lecture Line 2 ===
        # Multiple semi-transparent blue dots (#00FFFF) enter and explore simultaneously.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        q_color = "#00FFFF"
        qm1 = Dot(color=q_color, fill_opacity=0.4)
        qm2 = Dot(color=q_color, fill_opacity=0.4)
        qm3 = Dot(color=q_color, fill_opacity=0.4)
        
        # Issue 43 Fix: Starting at A2 with scale 1.2
        for qm in [qm1, qm2, qm3]: 
            self.place_at_grid(qm, "A2", scale_factor=1.2)
        self.add(qm1, qm2, qm3)
        
        # Parallel Movement logic
        self.play(
            qm1.animate.move_to(self.grid["A3"]), # Dead end path
            qm2.animate.move_to(self.grid["B2"]), # Path towards C2 dead end
            qm3.animate.move_to(self.grid["B2"]), # Path towards success
            run_time=1.5
        )
        
        self.play(
            qm2.animate.move_to(self.grid["C2"]),
            qm3.animate.move_to(self.grid["B3"]),
            run_time=0.8
        )

        # === Animation for Lecture Line 3 ===
        # One blue dot hits the exit star (#FFFF00), and all other ghostly paths fade away.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Path for success continues through the expanded grid space
        path_steps = ["B4", "B5", "B6", "C6", "D6", "E6", "F6"]
        for step in path_steps:
            self.play(qm3.animate.move_to(self.grid[step]), run_time=0.3)
        
        # Final success and collapse of superposition
        self.play(
            Flash(exit_star, color="#FFFF00"),
            qm3.animate.set_fill(opacity=1).scale(1.2),
            FadeOut(qm1),
            FadeOut(qm2),
            run_time=1
        )
        
        self.wait(2)
