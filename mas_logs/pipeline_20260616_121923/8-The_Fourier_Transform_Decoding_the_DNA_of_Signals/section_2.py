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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Prerequisite: The Building Blocks", 
            [
                "Basic sine waves are our fundamental building blocks.", 
                "They vary by their frequency and their amplitude.", 
                "Any complex wave is just a sum of these."
            ]
        )
        
        # Colors
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#5555FF"
        PURPLE_COLOR = "#A020F0"
        
        # Assets
        FORK_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/tuningfork.svg"

        # Value tracker for time-based animation
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED_COLOR))
        
        # Load SVG assets for forks
        fork1 = SVGMobject(FORK_ASSET, color=RED_COLOR)
        fork2 = SVGMobject(FORK_ASSET, color=GREEN_COLOR)
        fork3 = SVGMobject(FORK_ASSET, color=BLUE_COLOR)
        
        # Resolve Issue 38: Move to Column 2 (B2, C2, D2)
        self.place_at_grid(fork1, "B2", scale_factor=0.8)
        self.place_at_grid(fork2, "C2", scale_factor=0.8)
        self.place_at_grid(fork3, "D2", scale_factor=0.8)
        
        self.play(Create(fork1), Create(fork2), Create(fork3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN_COLOR))
        
        # Wave parameters: (amplitude, frequency)
        waves_config = [(0.3, 1), (0.2, 2), (0.1, 4)]
        colors = [RED_COLOR, GREEN_COLOR, BLUE_COLOR]
        forks = [fork1, fork2, fork3]
        grid_rows = ["B", "C", "D"]
        
        active_waves = VGroup()
        all_axes = VGroup()
        
        for i, (amp, freq) in enumerate(waves_config):
            # Create a localized axes for each wave, moved to Column 3-6 to avoid overlapping forks
            row = grid_rows[i]
            axes = Axes(
                x_range=[0, 4, 1], y_range=[-0.5, 0.5, 1], 
                x_length=3.0, y_length=0.8, 
                axis_config={"include_tip": False, "stroke_width": 1}
            )
            self.place_in_area(axes, f"{row}3", f"{row}6")
            all_axes.add(axes)
            
            # Sine wave function
            wave = always_redraw(lambda a=axes, am=amp, f=freq, c=colors[i]: a.plot(
                lambda x: am * np.sin(2 * np.pi * f * (x - time_tracker.get_value())),
                color=c
            ))
            
            # Stable vibration for the forks linked to their grid position
            target_pos = self.grid[f"{row}2"].copy()
            forks[i].add_updater(lambda m, tp=target_pos: m.move_to(tp + RIGHT * 0.05 * np.sin(25 * time_tracker.get_value())))
            
            self.add(axes)
            self.play(Create(wave), run_time=1)
            active_waves.add(wave)

        self.play(time_tracker.animate.set_value(2), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PURPLE_COLOR))
        
        # Create a single axes for the merged wave at the bottom
        sum_axes = Axes(
            x_range=[0, 4, 1], y_range=[-0.7, 0.7, 1], 
            x_length=4.0, y_length=1.2,
            axis_config={"include_tip": False, "stroke_width": 1}
        )
        # Resolve Issue 38: place_in_area with scale_factor=0.8 and starting at Col 2
        self.place_in_area(sum_axes, "E2", "F6", scale_factor=0.8)
        
        # Sum wave function
        def complex_wave_func(x):
            val = 0
            for amp, freq in waves_config:
                val += amp * np.sin(2 * np.pi * freq * (x - time_tracker.get_value()))
            return val

        complex_wave = always_redraw(lambda: sum_axes.plot(
            complex_wave_func, color=PURPLE_COLOR
        ))
        
        self.add(sum_axes)
        
        # Visual transition: individual axes fade and move toward the sum area
        self.play(
            all_axes.animate.match_y(sum_axes).set_opacity(0),
            Create(complex_wave),
            run_time=2
        )
        
        self.play(time_tracker.animate.set_value(5), run_time=4, rate_func=linear)
        self.wait(2)
