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
        # Define the lecture script and title
        title = "Dispersion: Why Rainbows Happen"
        lines = [
            "Because n varies, different colors travel at different speeds.",
            "Violet light interacts more strongly than red light.",
            "This causes violet to slow down and bend more.",
            "A prism separates white light into a vibrant rainbow.",
            "We call this frequency-dependent bending normal dispersion."
        ]
        self.setup_layout(title, lines)

        # Shared time tracker for synchronized oscillator updaters
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # --- PRE-SETUP GEOMETRY ---
        # Prism using grid coordinates
        prism = Polygon(
            self.grid['B3'] + UP*0.3, 
            self.grid['E1'], 
            self.grid['E5'], 
            color="#888888", 
            fill_opacity=0.2, 
            stroke_width=3
        )
        
        # Ray Anchor Points
        # 1. Incident
        p_start = self.grid['C1'] + LEFT * 1.5
        p_hit_1 = self.grid['D2'] + LEFT * 0.2
        
        # 2. Internal Split
        p_hit_red_2 = self.grid['C4'] + RIGHT * 0.2
        p_hit_violet_2 = self.grid['E4'] + RIGHT * 0.1
        
        # 3. Exit Spectrum
        p_out_red = self.grid['B6'] + RIGHT * 0.5
        p_out_violet = self.grid['F6'] + RIGHT * 0.5
        
        # Ray Objects
        white_ray = Line(p_start, p_hit_1, color=WHITE, stroke_width=4)
        
        red_internal = Line(p_hit_1, p_hit_red_2, color="#FF0000", stroke_width=3)
        violet_internal = Line(p_hit_1, p_hit_violet_2, color="#EE82EE", stroke_width=3)
        
        red_external = Line(p_hit_red_2, p_out_red, color="#FF0000", stroke_width=3)
        violet_external = Line(p_hit_violet_2, p_out_violet, color="#EE82EE", stroke_width=3)
        
        # Spring-Mass setup helper
        def create_oscillator(color, position, amplitude):
            base_line = Line(position + LEFT*0.2, position + RIGHT*0.2, color=color, stroke_width=1)
            spring = Line(position, position, color=color)
            mass = Dot(position, radius=0.06, color=color)
            group = VGroup(base_line, spring, mass)
            
            def oscillator_updater(obj):
                t = time_tracker.get_value()
                offset = amplitude * np.sin(t * 15)
                obj[2].move_to(position + UP * offset)
                obj[1].put_start_and_end_on(position, position + UP * offset)
                
            group.add_updater(oscillator_updater)
            return group

        osc_red = create_oscillator("#FF0000", self.grid['C3'], 0.15)
        osc_violet = create_oscillator("#EE82EE", self.grid['E3'], 0.4)
        
        # Labels - Positioned according to Issue 37 and 38
        label_red = Text("Faster", font_size=18, color="#FF0000")
        label_violet = Text("Slower", font_size=18, color="#EE82EE")
        self.place_at_grid(label_red, 'A5', scale_factor=0.8)
        self.place_at_grid(label_violet, 'F5', scale_factor=0.8)

        # Rainbow Spectrum lines (internal filler)
        colors = ["#FF7F00", "#FFFF00", "#00FF00", "#0000FF"] # Orange, Yellow, Green, Blue
        spectrum_out = VGroup()
        for i, color in enumerate(colors):
            alpha = (i + 1) / (len(colors) + 1)
            target_p2 = p_hit_red_2 * (1-alpha) + p_hit_violet_2 * alpha
            target_p_out = p_out_red * (1-alpha) + p_out_violet * alpha
            spectrum_out.add(Line(target_p2, target_p_out, color=color, stroke_width=2))

        # === Animation for Lecture Line 1 ===
        # "Because n varies, different colors travel at different speeds."
        self.lecture[0].set_color(YELLOW)
        self.play(Create(prism), run_time=1)
        self.play(Create(white_ray), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Violet light interacts more strongly than red light."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            Create(red_internal),
            Create(violet_internal),
            FadeIn(osc_red),
            FadeIn(osc_violet),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This causes violet to slow down and bend more."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            Write(label_red),
            Write(label_violet),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A prism separates white light into a vibrant rainbow."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(
            Create(red_external),
            Create(violet_external),
            Create(spectrum_out),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We call this frequency-dependent bending normal dispersion."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Subtle emphasis: pulse the oscillators
        self.play(
            osc_red.animate.scale(1.2),
            osc_violet.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
