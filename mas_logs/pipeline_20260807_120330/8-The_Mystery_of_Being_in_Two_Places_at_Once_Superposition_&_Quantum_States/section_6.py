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
        lecture_lines = [
            "Quantum states transition from coins to vectors.",
            "Superposition empowers quantum computers to solve mysteries.",
            "This technology will redefine the future of computing."
        ]
        self.setup_layout("Summary & Real-World Impact", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display icons for 'Spinning Coin' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg], 'Vector', and 'Bloch Sphere'.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Asset integration for coin (Issue 26)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg").set_color(YELLOW)
        vector = Arrow(start=LEFT*0.4, end=RIGHT*0.4, color=GREEN, buff=0)
        
        # Bloch Sphere Icon
        sphere_circle = Circle(radius=0.4, color=BLUE)
        sphere_ellipse = Ellipse(width=0.8, height=0.3, color=BLUE, stroke_width=2).set_stroke(opacity=0.5)
        sphere_axes = VGroup(Line(UP*0.4, DOWN*0.4), Line(LEFT*0.4, RIGHT*0.4)).set_stroke(color=BLUE, width=1)
        bloch_sphere = VGroup(sphere_circle, sphere_ellipse, sphere_axes)

        self.place_at_grid(coin, 'B2', scale_factor=0.6)
        self.place_at_grid(vector, 'B4')
        self.place_at_grid(bloch_sphere, 'B5', scale_factor=0.8) # Issue 40: B6 -> B5
        
        label_coin = Text("Coin", font_size=16, color=YELLOW).next_to(coin, DOWN, buff=0.2)
        label_vector = Text("Vector", font_size=16, color=GREEN).next_to(vector, DOWN, buff=0.2)
        label_sphere = Text("Bloch Sphere", font_size=16, color=BLUE).next_to(bloch_sphere, DOWN, buff=0.2)

        self.play(
            FadeIn(coin), Write(label_coin),
            Rotate(coin, angle=2*PI, axis=Y_AXIS, run_time=1.5)
        )
        self.play(FadeIn(vector), Write(label_vector))
        self.play(FadeIn(bloch_sphere), Write(label_sphere))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a classical robot navigating a maze path-by-path (color: #FF4500).
        # Show a 'Quantum Quarky' splitting into multiple versions (color: #00FFFF).
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(FadeOut(coin, vector, bloch_sphere, label_coin, label_vector, label_sphere))

        # Mazes (Adjusted to rows D-F to accommodate labels in Row C and improve spacing)
        classical_color = "#FF4500"
        quantum_color = "#00FFFF"
        
        # Classical Maze (D2 to F3)
        c_maze = VGroup(
            Line(self.grid['D2'], self.grid['D3'], color=classical_color),
            Line(self.grid['F2'], self.grid['F3'], color=classical_color),
            Line(self.grid['D2'], self.grid['F2'], color=classical_color),
            Line(self.grid['D3'], self.grid['F3'], color=classical_color),
            Line(self.grid['E2'], self.grid['E3'], color=classical_color) # internal horizontal wall
        )
        
        # Quantum Maze (D5 to F6)
        q_maze = VGroup(
            Line(self.grid['D5'], self.grid['D6'], color=quantum_color),
            Line(self.grid['F5'], self.grid['F6'], color=quantum_color),
            Line(self.grid['D5'], self.grid['F5'], color=quantum_color),
            Line(self.grid['D6'], self.grid['F6'], color=quantum_color),
            Line(self.grid['E5'], self.grid['E6'], color=quantum_color) # internal horizontal wall
        )
        
        label_classical = Text("Classical", font_size=18, color=classical_color).next_to(self.grid['D2'], UP, buff=0.3)
        label_quantum = Text("Quantum", font_size=18, color=quantum_color).next_to(self.grid['D5'], UP, buff=0.3)
        
        robot = Square(side_length=0.25, color=classical_color, fill_opacity=0.8)
        self.place_at_grid(robot, 'D2', scale_factor=0.8) # Issue 41: Moved to D2
        
        quarky = Circle(radius=0.12, color=quantum_color, fill_opacity=0.8)
        self.place_at_grid(quarky, 'D5', scale_factor=0.8) # Issue 42: Moved to D5
        
        self.play(Create(c_maze), Create(q_maze), Write(label_classical), Write(label_quantum))
        self.play(FadeIn(robot), FadeIn(quarky))
        
        # Classical Path 1 (Blocked)
        self.play(robot.animate.move_to(self.grid['D3']), run_time=0.4)
        self.play(robot.animate.move_to(self.grid['E3']), run_time=0.4) # Hits internal wall
        self.wait(0.2)
        self.play(robot.animate.move_to(self.grid['D3']), run_time=0.4)
        self.play(robot.animate.move_to(self.grid['D2']), run_time=0.4)
        
        # Quantum Split
        q1 = quarky.copy()
        q2 = quarky.copy()
        q3 = quarky.copy()
        self.add(q1, q2, q3)
        self.remove(quarky)
        
        # Movement: Classical starts second path, Quantum explore all
        self.play(
            robot.animate.move_to(self.grid['E2']),
            q1.animate.move_to(self.grid['D6']), # Path 1 start
            q2.animate.move_to(self.grid['E5']), # Path 2 start
            q3.animate.move_to(self.grid['D5']), # Stay/Jitter
            run_time=0.6
        )
        
        self.play(
            robot.animate.move_to(self.grid['F2']),
            q1.animate.move_to(self.grid['E6']), # Path 1 hit wall
            q2.animate.move_to(self.grid['F5']), # Path 2 continue
            run_time=0.6
        )

        # === Animation for Lecture Line 3 ===
        # One Quarky version reaches the exit instantly, and the others fade.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        self.play(
            robot.animate.move_to(self.grid['F3']), # Classical Finish
            q2.animate.move_to(self.grid['F6']), # Quantum Finish
            FadeOut(q1),
            FadeOut(q3),
            run_time=0.6
        )
        
        self.wait(2)
