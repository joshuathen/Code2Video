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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Population vs. Sample", [
            "Imagine a large population with diverse individual traits.",
            "Their weights follow a very strange, non-normal distribution.",
            "Scientist Sam arrives to study these mysterious squirrels."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Create population of 50 circles
        population = VGroup()
        for _ in range(50):
            gray_val = np.random.uniform(0.4, 0.9)
            color = interpolate_color(GRAY_E, WHITE, gray_val)
            circle = Circle(radius=0.08, color=color, fill_opacity=0.8, stroke_width=1)
            population.add(circle)
        
        # Distribute randomly in area B2 to E5
        tl = self.grid["B2"]
        br = self.grid["E5"]
        for dot in population:
            x = np.random.uniform(tl[0], br[0])
            y = np.random.uniform(br[1], tl[1])
            dot.move_to([x, y, 0])
            
        pop_label = Text("Population", font_size=24, color="#FFFFFF")
        self.place_at_grid(pop_label, "A3")
        
        self.play(FadeIn(population), FadeIn(pop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update colors: Line 1 stays White, Line 2 to Gold
        self.play(
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # U-shape distribution positions
        rows_list = ["A", "B", "C", "D", "E", "F"]
        u_positions = []
        # Column 2 (Left Peak): Rows B, C, D (15 dots)
        for i in range(15):
            row_idx = 1 + (i % 3)
            pos = self.grid[f"{rows_list[row_idx]}2"] + np.random.uniform(-0.3, 0.3, 3)
            u_positions.append(pos)
        # Column 5 (Right Peak): Rows B, C, D (15 dots)
        for i in range(15):
            row_idx = 1 + (i % 3)
            pos = self.grid[f"{rows_list[row_idx]}5"] + np.random.uniform(-0.3, 0.3, 3)
            u_positions.append(pos)
        # Column 3 & 4 (The Valley): Row D (20 dots)
        for i in range(20):
            col_idx = 3 + (i % 2) # 3, 4
            pos = self.grid[f"D{col_idx}"] + np.random.uniform(-0.3, 0.3, 3)
            u_positions.append(pos)

        move_anims = [population[i].animate.move_to(u_positions[i]) for i in range(50)]
            
        dist_label = Text("Non-Normal Distribution", font_size=24, color="#FFD700")
        # Fix for Issue 18: Move dist_label to area A2-A5 to avoid overlap/cramping
        self.place_in_area(dist_label, 'A2', 'A5', scale_factor=0.8)

        self.play(
            *move_anims,
            FadeOut(pop_label),
            FadeIn(dist_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update colors: Line 2 to White, Line 3 to Sky Blue
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#87CEEB")
        )
        
        # Scientist Sam (Stick figure)
        sam_color = "#87CEEB"
        head = Circle(radius=0.15, color=sam_color).shift(UP*0.15)
        body = Line(ORIGIN, DOWN*0.5, color=sam_color)
        arms = Line(LEFT*0.3, RIGHT*0.3, color=sam_color).shift(DOWN*0.15)
        leg_l = Line(DOWN*0.5, DOWN*0.8 + LEFT*0.2, color=sam_color)
        leg_r = Line(DOWN*0.5, DOWN*0.8 + RIGHT*0.2, color=sam_color)
        sam = VGroup(head, body, arms, leg_l, leg_r)
        
        # Fix for Issue 19: Move Sam to C2 to avoid left-side crowding
        self.place_at_grid(sam, "C2", scale_factor=1.2)
        
        sam_label = Text("Scientist Sam", font_size=20, color=sam_color)
        # Fix for Issue 20: Move sam_label to D2 with appropriate scaling
        self.place_at_grid(sam_label, "D2", scale_factor=0.8)

        self.play(FadeIn(sam), FadeIn(sam_label))
        self.wait(2)
