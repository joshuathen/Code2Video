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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Convergence", [
            "Partial products oscillate around pi over two.",
            "Each step narrows the gap.",
            "The value homes in exactly."
        ])
        
        # --- Elements ---
        target_val = PI / 2
        line = NumberLine(x_range=[1.2, 2.0, 0.1], length=4, include_numbers=True).rotate(90 * DEGREES)
        self.place_in_area(line, 'B3', 'E4', scale_factor=0.9)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(ruler, 'B6', scale_factor=0.5)
        self.place_at_grid(compass, 'E6', scale_factor=0.5)
        
        target_dot = Dot(color=YELLOW).move_to(line.n2p(target_val))
        label_pi_over_2 = Text("π/2", font_size=20, color=YELLOW)
        self.place_at_grid(label_pi_over_2, 'C4', scale_factor=0.7)
        label_pi_over_2.next_to(target_dot, RIGHT)
        
        point = Dot(color=WHITE, radius=0.1)
        self.place_at_grid(point, 'C3', scale_factor=0.8)
        point.move_to(line.n2p(1.3)) # Start off-target
        
        self.add(line, target_dot, label_pi_over_2, point, ruler, compass)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        path = VGroup()
        current_val = 1.3
        for i in range(1, 4):
            direction = 1 if i % 2 != 0 else -1
            current_val = target_val + direction * (0.5 / i)
            target_pos = line.n2p(current_val)
            self.play(point.animate.move_to(target_pos), run_time=0.5)
            
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        for i in range(4, 7):
            direction = 1 if i % 2 != 0 else -1
            current_val = target_val + direction * (0.2 / i)
            target_pos = line.n2p(current_val)
            self.play(point.animate.move_to(target_pos), run_time=0.4)
            
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        target_pos = line.n2p(target_val)
        self.play(point.animate.move_to(target_pos).set_color(RED), run_time=1.0)
        self.play(Flash(point, color=RED))
