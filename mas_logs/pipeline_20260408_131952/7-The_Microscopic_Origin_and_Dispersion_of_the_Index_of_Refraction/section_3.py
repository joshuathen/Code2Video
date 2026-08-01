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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Origin: Wave Superposition", 
            [
                'Vibrating electrons emit their own secondary electromagnetic waves.', 
                'These secondary waves lag slightly behind the original wave.', 
                'The total wave is the sum of both waves.', 
                'This superposition results in a phase-shifted net wave.', 
                'The delayed phase appears as a slower wave velocity.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#0000FF")
        
        # Initialize internal_time to avoid conflict with read-only Scene.time property
        self.internal_time = 0
        
        electron = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/electron.svg").set_color("#0000FF")
        self.place_at_grid(electron, 'C3', scale_factor=0.5)
        
        # Vibrate electron
        electron.add_updater(lambda m, dt: m.shift(UP * 0.1 * np.sin(self.internal_time * 10) * dt * 10))
        self.add(electron)
        
        # Concentric cyan circles
        circles = VGroup()
        def create_expanding_circle():
            c = Circle(radius=0.1, color="#00FFFF", stroke_width=2)
            c.move_to(electron.get_center())
            return c

        def update_circles(obj, dt):
            self.internal_time += dt
            for c in obj:
                c.scale(1 + 2 * dt)
                c.set_stroke(opacity=max(0, 1 - c.get_radius() / 3))
            
            # Periodically add a new circle
            if int(self.internal_time * 5) > int((self.internal_time - dt) * 5):
                obj.add(create_expanding_circle())
            
            # Remove old circles (using list to safely remove during iteration)
            for c in list(obj):
                if c.get_radius() > 3:
                    obj.remove(c)

        circles.add_updater(update_circles)
        self.add(circles)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(FadeOut(electron), FadeOut(circles))
        
        axes = Axes(
            x_range=[0, 6.28, 1], 
            y_range=[-2, 2, 1], 
            x_length=5, 
            y_length=3, 
            axis_config={"include_tip": False}
        ).set_color(GRAY)
        self.place_in_area(axes, 'B1', 'E6')
        self.add(axes)

        # Original white wave
        original_wave = axes.plot(lambda x: np.sin(2 * x), color="#FFFFFF")
        original_label = Text("Original", font_size=16, color="#FFFFFF")
        self.place_at_grid(original_label, 'B5')

        # Secondary cyan wave (shifted right/lagging)
        secondary_wave = axes.plot(lambda x: 0.6 * np.sin(2 * x - 1.2), color="#00FFFF")
        secondary_label = Text("Secondary", font_size=16, color="#00FFFF")
        self.place_at_grid(secondary_label, 'E5')

        self.play(Create(original_wave), Write(original_label))
        self.play(Create(secondary_wave), Write(secondary_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        # Vertical dashed lines to show summation
        sum_lines = VGroup()
        for x_val in np.linspace(0.5, 5.5, 10):
            p1 = axes.c2p(x_val, np.sin(2 * x_val))
            p2 = axes.c2p(x_val, 0.6 * np.sin(2 * x_val - 1.2))
            line = DashedLine(p1, p2, color=WHITE, stroke_width=2)
            sum_lines.add(line)
        
        self.play(Create(sum_lines))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        
        # Net wave (Yellow)
        net_wave = axes.plot(lambda x: 1.35 * np.sin(2 * x - 0.45), color="#FFFF00", stroke_width=6)
        net_label = Text("Net Wave", font_size=20, color="#FFFF00")
        self.place_at_grid(net_label, 'A4')

        self.play(
            FadeOut(sum_lines),
            original_wave.animate.set_stroke(opacity=0.3),
            secondary_wave.animate.set_stroke(opacity=0.3),
            Create(net_wave),
            Write(net_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        
        # Show comparison of velocities
        self.play(FadeOut(secondary_wave), FadeOut(secondary_label), FadeOut(original_label), FadeOut(net_label))
        
        # Split screen into two axes for comparison
        self.play(FadeOut(axes), FadeOut(original_wave), FadeOut(net_wave))
        
        axes_vac = Axes(x_range=[0, 10], y_range=[-1.5, 1.5], x_length=5, y_length=1.5).set_color(GRAY)
        axes_med = Axes(x_range=[0, 10], y_range=[-1.5, 1.5], x_length=5, y_length=1.5).set_color(GRAY)
        
        self.place_in_area(axes_vac, 'B1', 'C6')
        self.place_in_area(axes_med, 'E1', 'F6')
        
        label_vac = Text("Vacuum (c)", font_size=18, color=WHITE)
        label_med = Text("Medium (v < c)", font_size=18, color=YELLOW)
        self.place_at_grid(label_vac, 'B1', scale_factor=0.8).shift(RIGHT * 0.5)
        self.place_at_grid(label_med, 'E1', scale_factor=0.8).shift(RIGHT * 0.5)

        time_tracker = ValueTracker(0)
        
        # Vacuum wave: sin(k(x - ct)) -> sin(2x - 10t)
        wave_vac = always_redraw(lambda: axes_vac.plot(
            lambda x: np.sin(2 * x - 10 * time_tracker.get_value()), color=WHITE
        ))
        
        # Medium wave: sin(k(x - vt)) -> sin(2x - 6t)
        wave_med = always_redraw(lambda: axes_med.plot(
            lambda x: np.sin(2 * x - 6 * time_tracker.get_value()), color=YELLOW
        ))

        self.add(axes_vac, axes_med, label_vac, label_med, wave_vac, wave_med)
        
        # Add tracker dot to visualize speed
        dot_vac = always_redraw(lambda: Dot(axes_vac.c2p((np.pi/4 + 10 * time_tracker.get_value()) % 10, 1), color=WHITE))
        dot_med = always_redraw(lambda: Dot(axes_med.c2p((np.pi/4 + 6 * time_tracker.get_value()) % 10, 1), color=YELLOW))
        
        self.add(dot_vac, dot_med)
        
        self.play(time_tracker.animate.set_value(3), run_time=5, rate_func=linear)
        self.wait(2)
