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
        title_text = "The Smoothie Metaphor (The Hook)"
        lecture_lines = [
            "A smoothie is a blend of different ingredients.",
            "Complex signals are blends of simple sine waves.",
            "The Fourier Transform acts like a mathematical prism.",
            "It separates a combined signal into constituent parts.",
            "This reveals the hidden recipe of our reality."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Simple smoothie icon using shapes
        cup = Polygon(
            [-0.5, -0.8, 0], [0.5, -0.8, 0], [0.7, 0.8, 0], [-0.7, 0.8, 0],
            color=WHITE, stroke_width=2
        )
        liquid_bottom = Polygon(
            [-0.45, -0.7, 0], [0.45, -0.7, 0], [0.5, -0.1, 0], [-0.5, -0.1, 0],
            color="#FFFF00", fill_opacity=0.8, stroke_width=0
        )
        liquid_top = Polygon(
            [-0.5, -0.1, 0], [0.5, -0.1, 0], [0.65, 0.7, 0], [-0.65, 0.7, 0],
            color="#FFC0CB", fill_opacity=0.8, stroke_width=0
        )
        smoothie = VGroup(liquid_bottom, liquid_top, cup)
        
        self.place_in_area(smoothie, 'B3', 'E4', scale_factor=1.0)
        self.play(FadeIn(smoothie))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Split smoothie into ingredients
        banana = Circle(radius=0.2, color="#FFFFE0", fill_opacity=1)
        strawberry = Triangle(color="#FF9999", fill_opacity=1).scale(0.2)
        milk = RoundedRectangle(height=0.4, width=0.3, corner_radius=0.05, color="#FFFFFF", fill_opacity=1)
        
        banana_label = Text("Banana", font_size=14, color="#FFFFE0")
        strawberry_label = Text("Strawberry", font_size=14, color="#FF9999")
        milk_label = Text("Milk", font_size=14, color="#FFFFFF")
        
        banana_grp = VGroup(banana, banana_label.next_to(banana, DOWN, buff=0.1))
        strawberry_grp = VGroup(strawberry, strawberry_label.next_to(strawberry, DOWN, buff=0.1))
        milk_grp = VGroup(milk, milk_label.next_to(milk, DOWN, buff=0.1))
        
        ingredients = VGroup(banana_grp, strawberry_grp, milk_grp).arrange(RIGHT, buff=0.4)
        # Fix Issue 20: Position ingredients in A1-A6 to avoid overlap
        self.place_in_area(ingredients, 'A1', 'A6', scale_factor=1.0)
        
        # Complex wave
        axes_complex = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], x_length=4.5, y_length=1.5,
                           axis_config={"include_tip": False, "stroke_width": 1}).set_color(GRAY)
        self.place_in_area(axes_complex, 'B1', 'B6')
        
        complex_wave = axes_complex.plot(
            lambda x: 0.5*np.sin(2*PI*0.5*x) + 0.3*np.sin(2*PI*1.5*x) + 0.2*np.sin(2*PI*3*x),
            color=WHITE
        )
        
        self.play(
            FadeOut(smoothie),
            FadeIn(ingredients),
            Create(axes_complex),
            Create(complex_wave)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fix Issue 22: Prism scaled down and placed in area D3-D4
        prism = Polygon(
            [-0.5, -0.4, 0], [0.5, -0.4, 0], [0, 0.6, 0],
            color="#ADD8E6", fill_opacity=0.3, stroke_width=2
        )
        self.place_in_area(prism, 'D3', 'D4', scale_factor=0.7)
        
        self.play(FadeIn(prism))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Decomposed waves
        axes_pure = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], x_length=4.5, y_length=1.5,
                        axis_config={"include_tip": False, "stroke_width": 1}).set_color(GRAY)
        self.place_in_area(axes_pure, 'F1', 'F6')
        
        wave1 = axes_pure.plot(lambda x: 0.5*np.sin(2*PI*0.5*x), color="#FFFFE0")
        wave2 = axes_pure.plot(lambda x: 0.3*np.sin(2*PI*1.5*x), color="#FF9999")
        wave3 = axes_pure.plot(lambda x: 0.2*np.sin(2*PI*3*x), color="#FFFFFF")
        
        self.play(
            complex_wave.animate.move_to(prism.get_center()).scale(0.1).set_opacity(0),
            FadeIn(axes_pure),
            LaggedStart(Create(wave1), Create(wave2), Create(wave3), lag_ratio=0.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Fix Issue 21: Labels at C1-C2 and E1-E2
        time_label = Text("Time Domain", font_size=16, color=WHITE)
        freq_label = Text("Frequency Domain", font_size=16, color=WHITE)
        
        self.place_in_area(time_label, 'C1', 'C2', scale_factor=1.0)
        self.place_in_area(freq_label, 'E1', 'E2', scale_factor=1.0)
        
        self.play(
            FadeIn(time_label),
            FadeIn(freq_label)
        )
        self.wait(2)
