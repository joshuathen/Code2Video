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
        # Colors
        PURPLE = "#9B59B6"
        YELLOW = "#F1C40F"
        BLUE = "#3498DB"

        title = "The Smoothie Analogy (Introduction)"
        lines = [
            "Complex signals are like smoothies made of many ingredients.",
            "Individual ingredients are simple, repeating sine waves.",
            "Fourier Transforms reveal the exact recipe for any signal."
        ]
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Complex signals are like smoothies made of many ingredients.
        self.lecture[0].set_color(PURPLE)
        
        # Smoothie cup icon
        smoothie_cup = RoundedRectangle(
            corner_radius=0.2, 
            height=1.5, 
            width=1.0, 
            color=PURPLE, 
            fill_opacity=0.8
        )
        self.place_in_area(smoothie_cup, "C3", "D4")
        
        # Chaotic purple waveform
        # Critic Fix (Issue 47): Moved chaotic_wave to A2-A5 to avoid overlap with smoothie_cup
        chaotic_wave = FunctionGraph(
            lambda x: np.sin(2 * x) + 0.5 * np.sin(5 * x),
            x_range=[-2, 2],
            color=PURPLE
        )
        self.place_in_area(chaotic_wave, "A2", "A5", scale_factor=0.6)
        
        self.play(FadeIn(smoothie_cup), Create(chaotic_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Individual ingredients are simple, repeating sine waves.
        
        # Use Banana asset [Issue 43]
        banana_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/banana.svg"
        try:
            banana_icon = SVGMobject(banana_asset_path).set_color(YELLOW)
        except:
            # Fallback for local testing or missing files
            banana_icon = Circle(radius=0.3, color=YELLOW, fill_opacity=1)
            
        self.place_at_grid(banana_icon, "A1", scale_factor=0.6)
        
        # Blueberry icon (no specific asset path provided, using colored dot)
        blueberry_icon = Dot(color=BLUE, radius=0.2)
        self.place_at_grid(blueberry_icon, "A6", scale_factor=0.8)
        
        # Split waves
        # Critic Fix (Issue 48): yellow_wave moved to C1-C2 to avoid overlap
        yellow_wave = FunctionGraph(
            lambda x: np.sin(2 * x),
            x_range=[-1, 1],
            color=YELLOW
        )
        self.place_in_area(yellow_wave, "C1", "C2", scale_factor=0.5)
        
        blue_wave = FunctionGraph(
            lambda x: 0.5 * np.sin(5 * x),
            x_range=[-2, 2],
            color=BLUE
        )
        self.place_in_area(blue_wave, "E2", "E5", scale_factor=0.5)
        
        self.play(
            FadeIn(banana_icon, shift=RIGHT),
            FadeIn(blueberry_icon, shift=LEFT),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        self.play(
            ReplacementTransform(chaotic_wave.copy(), yellow_wave),
            ReplacementTransform(chaotic_wave.copy(), blue_wave),
            chaotic_wave.animate.set_stroke(opacity=0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fourier Transforms reveal the exact recipe for any signal.
        
        # Critic Fix (Issue 49): banana_label moved to B1 to avoid overlap
        banana_label = Text("Banana: 440Hz", font_size=16, color=YELLOW)
        self.place_at_grid(banana_label, "B1")
        
        blueberry_label = Text("Blueberry: 523Hz", font_size=16, color=BLUE)
        self.place_at_grid(blueberry_label, "F2")
        
        self.play(
            Write(banana_label),
            Write(blueberry_label),
            self.lecture[2].animate.set_color(PURPLE)
        )
        self.wait(2)
