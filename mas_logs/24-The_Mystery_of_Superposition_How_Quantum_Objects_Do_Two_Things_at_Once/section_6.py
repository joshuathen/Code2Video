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
        # Initial Setup
        title_str = "Application: Why Superposition Powers the Future"
        lines = [
            "Superposition allows computers to process data in parallel.",
            "A quantum mouse explores every path at once.",
            "This massive speedup will transform our digital future."
        ]
        self.setup_layout(title_str, lines)
        
        # Reference for grid dictionary
        G = self.grid

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Draw Maze Walls
        # Outer Boundary (A1 to F6)
        boundary = VGroup(
            Line(G["A1"] + UL*0.45, G["A6"] + UR*0.45), # Top
            Line(G["A6"] + UR*0.45, G["F6"] + DR*0.45), # Right
            Line(G["F6"] + DR*0.45, G["F1"] + DL*0.45), # Bottom
            Line(G["F1"] + DL*0.45, G["A1"] + UL*0.45)  # Left
        ).set_color(GRAY_C).set_stroke(width=4)
        
        # Internal Obstacles
        int_walls = VGroup(
            Line(G["B1"] + DL*0.45, G["B5"] + DR*0.45), # Horizontal barrier
            Line(G["A2"] + UR*0.45, G["D2"] + DR*0.45), # Vertical barrier
            Line(G["D4"] + UL*0.45, G["D6"] + UR*0.45)  # Another horizontal
        ).set_color(GRAY_C).set_stroke(width=4)
        
        maze = VGroup(boundary, int_walls)
        
        # Adjusted START position to avoid left margin (Issue 51)
        start_label = Text("START", font_size=16, color=WHITE).next_to(G["A2"], UP, buff=0.2)
        exit_label = Text("EXIT", font_size=16, color=WHITE).next_to(G["F6"], RIGHT, buff=0.3)
        
        self.play(Create(maze), FadeIn(start_label), FadeIn(exit_label))
        
        # Classical Mouse (White Dot)
        classical_mouse = Dot(color="#FFFFFF", radius=0.15)
        # Fixed: Move classical_mouse to A2 and scale to 0.6 (Issue 50)
        self.place_at_grid(classical_mouse, "A2", scale_factor=0.6)
        
        self.play(FadeIn(classical_mouse))
        
        # Simulation: Try one path, fail, backtrack
        # Tries to move into Wall 2 (Vertical at col 2-3)
        self.play(classical_mouse.animate.move_to(G["A3"]), run_time=0.6)
        self.play(classical_mouse.animate.scale(1.2).set_color(RED), run_time=0.1)
        self.play(classical_mouse.animate.scale(1/1.2).set_color(WHITE), run_time=0.1)
        self.play(classical_mouse.animate.move_to(G["A2"]), run_time=0.6)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Quantum Mouse (Cyan Dot)
        quantum_mouse_start = Dot(color="#00FFFF", radius=0.15)
        # Fixed: Move quantum_mouse_start to A2 and scale to 0.6 (Issue 51)
        self.place_at_grid(quantum_mouse_start, "A2", scale_factor=0.6)
        
        self.play(FadeOut(classical_mouse), FadeIn(quantum_mouse_start))
        
        # Split into superposition (Parallel paths)
        q_copy1 = quantum_mouse_start.copy() # Path A
        q_copy2 = quantum_mouse_start.copy() # Path B
        q_copy3 = quantum_mouse_start.copy() # Success path
        
        self.add(q_copy1, q_copy2, q_copy3)
        self.remove(quantum_mouse_start)
        
        # Simultaneous exploration from A2
        self.play(
            q_copy1.animate.move_to(G["A1"]),
            q_copy2.animate.move_to(G["B2"]),
            q_copy3.animate.move_to(G["A3"]),
            run_time=1.2
        )
        
        self.play(
            q_copy3.animate.move_to(G["F3"]),
            run_time=1.0
        )
        
        self.play(
            q_copy3.animate.move_to(G["F6"]),
            run_time=1.0
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Exit found!
        success_flash = Circle(radius=0.2, color="#FFFF00").move_to(G["F6"])
        
        self.play(
            q_copy3.animate.set_color("#FFFF00").scale(1.8),
            FadeIn(success_flash),
            q_copy1.animate.set_opacity(0),
            q_copy2.animate.set_opacity(0),
        )
        
        self.play(
            success_flash.animate.scale(3).set_opacity(0),
            run_time=0.8
        )
        
        self.wait(2)
