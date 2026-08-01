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
        # Define content
        title_text = "Prerequisite: The Ingredients of Statistics"
        lecture_lines = [
            'A population includes every single item in existence.',
            'A sample is just a small, random subset.',
            'We calculate the average, or mean, of each sample.'
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        POP_COLOR = "#FFA726"
        SAM_COLOR = "#29B6F6"
        MEAN_COLOR = "#FFEE58"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(POP_COLOR))
        
        # Population Container (Vat)
        # Using a RoundedRectangle to represent the "vat"
        pop_vat = RoundedRectangle(height=4.5, width=5.5, corner_radius=0.4, color=POP_COLOR, fill_opacity=0.1)
        # Place it to occupy top 5 rows to avoid overlapping bottom elements (Fix Issue 36)
        self.place_in_area(pop_vat, "A1", "E6")
        
        pop_label = Text("Population (Vat)", font_size=20, color=POP_COLOR)
        # Position label at top-left grid point
        self.place_at_grid(pop_label, "A1", scale_factor=0.8)
        
        # Fill with dots to represent individual items
        np.random.seed(0) # Deterministic for consistent layout
        pop_dots = VGroup(*[
            Dot(radius=0.03, color=POP_COLOR, fill_opacity=0.5).move_to(
                pop_vat.get_center() + np.array([np.random.uniform(-2.5, 2.5), np.random.uniform(-2.0, 2.0), 0])
            ) for _ in range(120)
        ])
        
        self.play(Create(pop_vat), Write(pop_label))
        self.play(FadeIn(pop_dots, lag_ratio=0.01))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(SAM_COLOR)
        )
        
        # Small scoop circle to pick up a subset from the population
        small_scoop = Circle(radius=0.4, color=SAM_COLOR, stroke_width=4)
        self.place_at_grid(small_scoop, "C3")
        
        # Zoom target (zoomed view of the sample subset)
        zoomed_scoop = Circle(radius=1.8, color=SAM_COLOR, fill_opacity=0.1, stroke_width=6)
        self.place_in_area(zoomed_scoop, "B2", "E5")
        
        # Visual transition: dim the population and expand the scoop
        self.play(
            Create(small_scoop),
            pop_vat.animate.set_stroke(opacity=0.1),
            pop_dots.animate.set_fill(opacity=0.1),
            pop_label.animate.set_fill(opacity=0.2)
        )
        
        self.play(
            ReplacementTransform(small_scoop, zoomed_scoop)
        )
        
        # Sample label for zoomed view
        scoop_label = Text("Sample Scoop (n=30)", font_size=22, color=SAM_COLOR)
        # Repositioned to avoid overlap (Fix Issue 35)
        self.place_in_area(scoop_label, 'A2', 'A4', scale_factor=0.8)
        self.play(Write(scoop_label))

        # Create 30 candy icons (hexagons with numeric values)
        candies = VGroup()
        values = [np.random.randint(1, 20) for _ in range(30)]
        for i in range(30):
            # Hexagon for candy representation
            candy_bg = RegularPolygon(n=6, radius=0.20, color=SAM_COLOR, fill_opacity=0.9)
            val_txt = Text(str(values[i]), font_size=12, color=WHITE)
            candy = VGroup(candy_bg, val_txt)
            
            # Arrange in 5 rows x 6 columns grid inside the zoomed scoop
            row = i // 6
            col = i % 6
            # Calculate local offset from the scoop center
            offset = np.array([(col - 2.5) * 0.5, (2 - row) * 0.5, 0])
            candy.move_to(zoomed_scoop.get_center() + offset)
            candies.add(candy)
            
        self.play(LaggedStartMap(FadeIn, candies, shift=UP*0.1, lag_ratio=0.03))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(MEAN_COLOR)
        )
        
        # Sample Mean formula using Text
        formula = Text("x̄ = Σx / n", color=MEAN_COLOR)
        # Shifted right to avoid lecture notes (Fix Issue 37)
        self.place_in_area(formula, "F2", "F3", scale_factor=1.1)
        
        # Display the specific calculated result for this sample
        mean_val = sum(values) / 30
        mean_display = Text(f"Mean (x̄) = {mean_val:.1f}", font_size=24, color=MEAN_COLOR)
        self.place_in_area(mean_display, "F4", "F6", scale_factor=1.0)
        
        # Animation: fade candies into mean label as per description
        self.play(
            Write(formula),
            FadeIn(mean_display),
            candies.animate.set_opacity(0.1),
            run_time=1.5
        )
        self.wait(2)
