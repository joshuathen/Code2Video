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

class Section5Scene(TeachingScene):
    def construct(self):
        # Fetching latest lecture lines from storyboard
        lecture_lines = [
            "Repeat these steps approximately square root N times.",
            "The target amplitude becomes nearly 100 percent.",
            "Measuring the system reveals the needle immediately."
        ]
        self.setup_layout("Iteration and Measurement", lecture_lines)
        
        # Colors based on storyboard and constraints
        COLOR_REPEAT = "#AAAAAA"
        COLOR_TARGET = "#FFFF00"
        COLOR_SUCCESS = "#00FF00"
        COLOR_DEFAULT = "#444444"
        COLOR_MEASURE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Repeat icon (circular 'repeat' icon in light grey #AAAAAA)
        repeat_icon = Arc(radius=0.5, start_angle=0, angle=1.5*PI, color=COLOR_REPEAT)
        repeat_icon.add_tip()
        # Issue 42: Repositioned to B5, scale 0.8 to avoid title area crowding
        self.place_at_grid(repeat_icon, "B5", scale_factor=0.8)
        
        # Bars setup (8 bars)
        bars = VGroup()
        target_index = 4 # Representing Safe #5 (index 4)
        num_bars = 8
        bar_width = 0.3
        
        for i in range(num_bars):
            h = 0.4 if i != target_index else 0.8
            c = COLOR_DEFAULT if i != target_index else COLOR_TARGET
            bar = Rectangle(width=bar_width, height=h, fill_opacity=0.8, stroke_width=1, color=c, fill_color=c)
            bars.add(bar)
        
        # Arrange bars in area E1 to F6
        bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        # Issue 42: Adjust 'bars' position to E1-F6 for vertical growth space
        self.place_in_area(bars, "E1", "F6", scale_factor=1.0)
        
        # Initial reveal and lecture color change
        self.play(self.lecture[0].animate.set_color(COLOR_REPEAT))
        self.play(Create(repeat_icon), Create(bars))
        
        # Yellow target bar pulses and grows taller while repeat icon rotates
        target_bar = bars[target_index]
        self.play(
            Rotate(repeat_icon, angle=-2*PI, about_point=repeat_icon.get_center()),
            target_bar.animate.scale(1.3, about_edge=DOWN),
            run_time=2, rate_func=there_and_back
        )

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_TARGET))
        
        # Target bar reaching near-maximum height while all other bars shrink to nearly zero.
        target_final_height = 3.5 # Increased height range
        others_final_height = 0.05
        
        anim_group = []
        for i, bar in enumerate(bars):
            new_h = target_final_height if i == target_index else others_final_height
            anim_group.append(bar.animate.stretch_to_fit_height(new_h, about_edge=DOWN))
        
        self.play(
            *anim_group,
            FadeOut(repeat_icon),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_SUCCESS))
        
        # Measurement meter icon [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/meter.svg]
        # Issue 42: Integrate asset, scale 0.7, position B3 to prevent overlap
        meter = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/meter.svg")
        meter.set_color(COLOR_MEASURE)
        self.place_at_grid(meter, "B3", scale_factor=0.7)
        
        self.play(FadeIn(meter))
        
        # Subtle "measuring" animation for the meter (shake)
        self.play(meter.animate.shift(LEFT*0.1), run_time=0.1)
        self.play(meter.animate.shift(RIGHT*0.2), run_time=0.1)
        self.play(meter.animate.shift(LEFT*0.1), run_time=0.1)
        
        # All bars vanish except target which turns solid green (#00FF00)
        vanish_anims = [FadeOut(bars[i]) for i in range(num_bars) if i != target_index]
        
        self.play(
            *vanish_anims,
            target_bar.animate.set_color(COLOR_SUCCESS).set_fill(COLOR_SUCCESS),
            FadeOut(meter),
            run_time=1.5
        )
        
        self.wait(3)
