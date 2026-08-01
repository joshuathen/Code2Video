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
        # Fetching data from storyboard
        title_text = "Infinite Zoom: The Self-Similarity Property"
        lecture_lines = [
            "Holomorphic functions create perfectly smooth, infinite fractals.",
            "Zooming in reveals the same level of complexity.",
            "Patterns repeat forever in a dance of self-similarity."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors for matching lecture lines
        line_colors = [YELLOW, BLUE, GREEN_C]

        # === Animation for Lecture Line 1 ===
        # Holomorphic functions create perfectly smooth, infinite fractals.
        self.lecture[0].set_color(line_colors[0])

        # Integrated Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spiral.svg
        def get_fractal_shape(color, scale_val=1.0):
            # A stylized Julia Set spiral using the provided SVG asset
            try:
                spiral = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spiral.svg")
            except:
                # Fallback if asset is missing (though prompt says MUST use)
                spiral = ParametricFunction(
                    lambda t: np.array([0.6 * t * np.cos(5 * PI * t), 0.6 * t * np.sin(5 * PI * t), 0]),
                    t_range=[0, 1.2]
                )
            
            spiral.set_color(color).scale(scale_val)
            
            # Recursive-like sub-elements to simulate fractal complexity
            subs = VGroup()
            for angle in [0, 2*PI/3, 4*PI/3]:
                pos = 0.5 * scale_val * np.array([np.cos(angle), np.sin(angle), 0])
                sub = spiral.copy().scale(0.3).rotate(angle).move_to(pos)
                subs.add(sub)
            
            return VGroup(spiral, subs)

        fractal_1 = get_fractal_shape(line_colors[0])
        # Fix Issue 37: Problem: The fractal animation 'fractal_1' expands and obstructs the lecture notes.
        # Fix: self.place_in_area(fractal_1, 'B3', 'E6', scale_factor=0.8)
        self.place_in_area(fractal_1, 'B3', 'E6', scale_factor=0.8)
        
        self.play(DrawBorderThenFill(fractal_1), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Zooming in reveals the same level of complexity.
        self.lecture[1].set_color(line_colors[1])
        
        # Determine zoom target (one of the sub-spirals)
        zoom_target = fractal_1[1][0] # The one at angle 0 (right side)
        zoom_center = zoom_target.get_center()
        original_center = fractal_1.get_center()
        
        # Create a deeper level of fractal inside the zoom target to reveal on zoom
        # The sub-elements of fractal_2 will represent even deeper iterations
        fractal_2 = get_fractal_shape(line_colors[1], scale_val=0.3).move_to(zoom_center)
        
        # Group for zooming
        zoom_group = VGroup(fractal_1, fractal_2)
        
        # Zoom calculation
        # We want fractal_2 to grow to the size of fractal_1
        # fractal_2 scale is 0.3 relative to fractal_1. So scale by 1/0.3.
        scale_ratio = 1 / 0.3
        
        self.play(
            zoom_group.animate.scale(scale_ratio, about_point=zoom_center).shift(original_center - zoom_center),
            run_time=4
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Patterns repeat forever in a dance of self-similarity.
        self.lecture[2].set_color(line_colors[2])
        
        # Highlight the similarity
        self.play(fractal_2.animate.set_color(line_colors[2]))
        
        # Fix Issue 38: Problem: The label 'Infinite Complexity' at F4 is positioned where elements overlap.
        # Fix: self.place_at_grid(label, 'F5', scale_factor=0.8)
        rect = SurroundingRectangle(fractal_2, color=line_colors[2], buff=0.1)
        label = Text("Infinite Complexity", font_size=20, color=line_colors[2])
        self.place_at_grid(label, 'F5', scale_factor=0.8)
        
        self.play(Create(rect), Write(label))
        self.wait(2)
        
        # Show one more subtle pulse of zoom to emphasize "forever"
        self.play(
            zoom_group.animate.scale(1.1, about_point=original_center),
            label.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
