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
        lecture_lines_text = [
            'A smoothie is a complex blend of many ingredients.', 
            'We need to know exactly what is inside.', 
            'The Fourier Transform acts like a smart filter.', 
            'It separates the blend into individual ingredient jars.', 
            'Any complex signal can be deconstructed this way.'
        ]
        self.setup_layout("The Smoothie Analogy: Why We Need It", lecture_lines_text)

        # Colors
        PURPLE_SMOOTHIE = "#A020F0"
        RED_STRAWBERRY = "#FF0000"
        YELLOW_BANANA = "#FFFF00"
        WHITE_MILK = "#FFFFFF"

        # Asset Paths
        GLASS_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/glass.svg"
        STRAWBERRY_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/strawberry.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(PURPLE_SMOOTHIE)
        
        # Glass outline using SVG asset
        glass_outline = SVGMobject(GLASS_ASSET, color=WHITE).set_stroke(color=WHITE, width=2)
        self.place_in_area(glass_outline, 'B3', 'E4', scale_factor=1.5)
        
        # Smoothie fill growing
        smoothie_fill = Rectangle(width=1.0, height=3.0, fill_color=PURPLE_SMOOTHIE, fill_opacity=0.8, stroke_width=0)
        self.place_in_area(smoothie_fill, 'B3', 'E4')
        
        smoothie_fill.save_state()
        smoothie_fill.stretch_to_fit_height(0.01)
        # Re-align to the bottom of the area B3-E4 (Row E is bottom)
        bottom_pos = (self.grid['E3'] + self.grid['E4']) / 2
        smoothie_fill.move_to(bottom_pos, aligned_edge=DOWN)
        
        self.play(Create(glass_outline))
        self.play(Restore(smoothie_fill), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        
        # Split into horizontal rectangles
        segment_h = 3.0 / 3
        strawberry_seg = Rectangle(width=1.0, height=segment_h, fill_color=RED_STRAWBERRY, fill_opacity=0.9, stroke_width=0)
        banana_seg = Rectangle(width=1.0, height=segment_h, fill_color=YELLOW_BANANA, fill_opacity=0.9, stroke_width=0)
        milk_seg = Rectangle(width=1.0, height=segment_h, fill_color=WHITE_MILK, fill_opacity=0.9, stroke_width=0)
        
        ingredients = VGroup(strawberry_seg, banana_seg, milk_seg).arrange(UP, buff=0)
        self.place_in_area(ingredients, 'B3', 'E4')
        
        self.play(ReplacementTransform(smoothie_fill, ingredients), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW_BANANA)
        
        # Label with Strawberry Asset
        strawberry_icon = SVGMobject(STRAWBERRY_ASSET).scale(0.3)
        label_s_text = Text("Strawberry", font_size=20, color=RED_STRAWBERRY)
        label_s = VGroup(strawberry_icon, label_s_text).arrange(RIGHT, buff=0.1)
        
        label_b = Text("Banana", font_size=20, color=YELLOW_BANANA)
        label_m = Text("Milk", font_size=20, color=WHITE_MILK)
        
        # Position labels at Col 5 (within 1 unit of the glass at Col 4)
        self.place_at_grid(label_s, 'B5', scale_factor=1.0)
        self.place_at_grid(label_b, 'C5', scale_factor=1.0)
        self.place_at_grid(label_m, 'D5', scale_factor=1.0)
        
        labels = VGroup(label_s, label_b, label_m)
        self.play(FadeIn(labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        
        # Scanning line across the glass
        scan_line = Line(
            start=self.grid['B2'], 
            end=self.grid['E2'], 
            color=WHITE, stroke_width=6
        )
        
        self.play(Create(scan_line))
        # Scan from Col 2 to Col 5
        self.play(scan_line.animate.move_to(self.grid['B5'], aligned_edge=UP), run_time=1.5)
        self.play(FadeOut(scan_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # Deconstructed jars in a row (Row F) to avoid occlusion and clutter
        jar_s = Rectangle(width=0.8, height=1.0, fill_color=RED_STRAWBERRY, fill_opacity=1, stroke_color=WHITE)
        jar_b = Rectangle(width=0.8, height=1.0, fill_color=YELLOW_BANANA, fill_opacity=1, stroke_color=WHITE)
        jar_m = Rectangle(width=0.8, height=1.0, fill_color=WHITE_MILK, fill_opacity=1, stroke_color=WHITE)
        
        # Positioning from Issues 29, 30, 31
        self.place_at_grid(jar_s, 'F2', scale_factor=0.8)
        self.place_at_grid(jar_b, 'F4', scale_factor=0.8)
        self.place_at_grid(jar_m, 'F6', scale_factor=0.8)
        
        jars = VGroup(jar_s, jar_b, jar_m)
        
        self.play(
            FadeOut(glass_outline),
            FadeOut(labels),
            ReplacementTransform(ingredients, jars),
            run_time=2
        )
        self.wait(2)
