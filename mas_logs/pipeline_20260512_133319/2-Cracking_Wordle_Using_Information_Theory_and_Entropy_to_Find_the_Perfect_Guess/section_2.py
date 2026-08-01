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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: The 'Bit' as a Unit of Information", 
            [
                "Information is measured in units called bits.", 
                "Start with a row of possible options.", 
                "One bit of information cuts your uncertainty in half.", 
                "Ten bits narrow down over one thousand possibilities.", 
                "Every choice leads you closer to the single solution."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        bit_text = Text("1 BIT", weight="BOLD", color="#FFFFFF")
        # Issue 38 Fix: Positioning '1 BIT' in row B
        self.place_in_area(bit_text, 'B3', 'B4', scale_factor=1.2)
        self.play(Write(bit_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        circles = VGroup(*[Circle(radius=0.15, color="#FFFFFF", stroke_width=2) for _ in range(8)])
        circles.arrange(RIGHT, buff=0.2)
        # Issues 39 & 40 Fix: Position circles at D2-D5 with scaling to avoid crowding
        self.place_in_area(circles, 'D2', 'D5', scale_factor=0.9)
        self.play(Create(circles))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        divider = Line(
            circles.get_top() + UP * 0.3,
            circles.get_bottom() + DOWN * 0.3,
            color="#FFFFFF",
            stroke_width=2
        )
        self.play(Create(divider))
        self.play(
            *[circles[i].animate.set_color("#404040") for i in range(4, 8)]
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeOut(circles), FadeOut(divider), FadeOut(bit_text))
        
        # Binary Tree Visualization
        root_pos = (self.grid['A3'] + self.grid['A4']) / 2
        tree_height = 4.0
        tree_width = 5.0
        levels = 10
        
        all_tree_lines = VGroup()
        path_indices = [0]
        current_idx = 0
        for l in range(levels):
            step = 1 if l % 2 == 0 else 0
            current_idx = current_idx * 2 + step
            path_indices.append(current_idx)
            
        path_lines = VGroup()
        
        for l in range(levels):
            num_nodes = 2**l
            y_curr = root_pos[1] - l * (tree_height / levels)
            y_next = root_pos[1] - (l + 1) * (tree_height / levels)
            
            for i in range(num_nodes):
                x_curr = root_pos[0] + (i - (num_nodes - 1) / 2) * (tree_width / num_nodes) if num_nodes > 1 else root_pos[0]
                
                for step in [0, 1]:
                    child_idx = i * 2 + step
                    num_next = 2**(l+1)
                    x_next = root_pos[0] + (child_idx - (num_next - 1) / 2) * (tree_width / num_next)
                    
                    line = Line(
                        [x_curr, y_curr, 0], 
                        [x_next, y_next, 0], 
                        stroke_width=max(0.3, 1.5 - l*0.15), 
                        color="#FFFFFF"
                    )
                    
                    if i == path_indices[l] and child_idx == path_indices[l+1]:
                        p_line = line.copy().set_color("#FFFF00").set_stroke(width=3)
                        path_lines.add(p_line)
                    
                    # Optimization: Sample branches at deeper levels to avoid performance hit
                    if l < 5:
                        all_tree_lines.add(line)
                    elif l < 7 and i % 2 == 0:
                        all_tree_lines.add(line)
                    elif l < 9 and i % 8 == 0:
                        all_tree_lines.add(line)
                    elif l == 9 and i % 32 == 0:
                        all_tree_lines.add(line)

        y_bottom = root_pos[1] - tree_height
        leaf_dots = VGroup()
        for i in range(1024):
            if i % 16 == 0:
                x = root_pos[0] + (i - 1023 / 2) * (tree_width / 1024)
                leaf_dots.add(Dot([x, y_bottom, 0], radius=0.015, color="#FFFFFF"))

        self.play(FadeIn(all_tree_lines), FadeIn(leaf_dots), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Trace path through the levels
        for branch in path_lines:
            self.play(Create(branch), run_time=0.2)
            
        final_dot = Dot(path_lines[-1].get_end(), radius=0.08, color="#FFFF00")
        self.play(Flash(final_dot, color="#FFFF00"), FadeIn(final_dot))
        self.wait(2)
        self.lecture[4].set_color(WHITE)
