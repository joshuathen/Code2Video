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
        title_str = "Orthogonality: How to Pick the Ingredients"
        lecture_lines = [
            "Orthogonality acts as a mathematical filter for specific frequencies.",
            "Multiply the signal by a target sine or cosine.",
            "Integrating over one period cancels out non-matching waves.",
            "Only the component matching the filter's frequency remains.",
            "This process isolates every individual ingredient in the recipe."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        COLOR_COMPLEX = "#00FFFF"  # Cyan
        COLOR_FILTER = "#FF00FF"   # Magenta
        COLOR_PRODUCT = "#FFFFFF"  # White
        COLOR_INTEGRAL = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Complex wave: f(x) = sin(x) + 0.5*sin(3x)
        complex_axes = Axes(
            x_range=[0, 2*np.pi], 
            y_range=[-1.5, 1.5], 
            axis_config={"include_tip": False, "color": GREY}
        ).scale(0.3)
        complex_wave = complex_axes.plot(lambda x: np.sin(x) + 0.5*np.sin(3*x), color=COLOR_COMPLEX)
        complex_label = Text("Complex Signal", font_size=16, color=COLOR_COMPLEX)
        
        complex_group = VGroup(complex_axes, complex_wave, complex_label).arrange(DOWN, buff=0.1)
        # Reduced scale_factor from 0.8 to 0.7 to improve spacing (Issue 28)
        self.place_in_area(complex_group, 'A1', 'B6', scale_factor=0.7)
        
        # Filter wave: g(x) = sin(x) (The target frequency)
        filter_axes = Axes(
            x_range=[0, 2*np.pi], 
            y_range=[-1.2, 1.2], 
            axis_config={"include_tip": False, "color": GREY}
        ).scale(0.3)
        filter_wave = filter_axes.plot(lambda x: np.sin(x), color=COLOR_FILTER)
        filter_label = Text("Filter Wave (sin(t))", font_size=16, color=COLOR_FILTER)
        
        filter_group = VGroup(filter_axes, filter_wave, filter_label).arrange(DOWN, buff=0.1)
        # Reduced scale_factor from 0.8 to 0.7 to improve spacing and legibility (Issue 29)
        self.place_in_area(filter_group, 'C1', 'D6', scale_factor=0.7)

        self.play(Create(complex_group), Create(filter_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Product wave: h(x) = signal(x) * filter(x)
        product_axes = Axes(
            x_range=[0, 2*np.pi], 
            y_range=[-1.0, 2.0], 
            axis_config={"include_tip": False, "color": GREY}
        ).scale(0.3)
        product_wave = product_axes.plot(
            lambda x: (np.sin(x) + 0.5*np.sin(3*x)) * np.sin(x), 
            color=COLOR_PRODUCT
        )
        product_label = Text("Product (Signal × Filter)", font_size=16, color=COLOR_PRODUCT)
        
        product_group = VGroup(product_axes, product_wave, product_label).arrange(DOWN, buff=0.1)
        # Reduced scale_factor from 0.8 to 0.7 to prevent cutoff at frame boundary (Issue 30)
        self.place_in_area(product_group, 'E1', 'F6', scale_factor=0.7)
        
        self.play(Create(product_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Area under the product wave represents the integral over one period
        area = product_axes.get_area(product_wave, x_range=[0, 2*np.pi], color=COLOR_INTEGRAL, opacity=0.5)
        
        self.play(FadeIn(area), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Show what happens with a non-matching frequency (sin(2t))
        mismatch_filter_wave = filter_axes.plot(lambda x: np.sin(2*x), color=COLOR_FILTER)
        mismatch_filter_label = Text("Filter Wave (sin(2t))", font_size=16, color=COLOR_FILTER)
        mismatch_filter_label.move_to(filter_label.get_center())
        
        mismatch_product_wave = product_axes.plot(
            lambda x: (np.sin(x) + 0.5*np.sin(3*x)) * np.sin(2*x), 
            color=COLOR_PRODUCT
        )
        mismatch_area = product_axes.get_area(mismatch_product_wave, x_range=[0, 2*np.pi], color=COLOR_INTEGRAL, opacity=0.5)
        
        # Transform to mismatch
        self.play(
            Transform(filter_wave, mismatch_filter_wave),
            Transform(filter_label, mismatch_filter_label),
            Transform(product_wave, mismatch_product_wave),
            Transform(area, mismatch_area),
            run_time=2
        )
        self.wait(1)
        
        # Revert back to matching frequency to show survival
        original_filter_wave = filter_axes.plot(lambda x: np.sin(x), color=COLOR_FILTER)
        original_filter_label = Text("Filter Wave (sin(t))", font_size=16, color=COLOR_FILTER)
        original_filter_label.move_to(filter_label.get_center())
        original_product_wave = product_axes.plot(
            lambda x: (np.sin(x) + 0.5*np.sin(3*x)) * np.sin(x), 
            color=COLOR_PRODUCT
        )
        original_area = product_axes.get_area(original_product_wave, x_range=[0, 2*np.pi], color=COLOR_INTEGRAL, opacity=0.5)

        self.play(
            Transform(filter_wave, original_filter_wave),
            Transform(filter_label, original_filter_label),
            Transform(product_wave, original_product_wave),
            Transform(area, original_area),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Emphasize the isolated component
        isolated_comp = complex_axes.plot(lambda x: np.sin(x), color=COLOR_COMPLEX).set_stroke(width=8)
        self.play(Indicate(isolated_comp, color=COLOR_COMPLEX), run_time=2)
        
        self.wait(2)
