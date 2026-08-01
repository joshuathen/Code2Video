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
        self.setup_layout("Prerequisite: The Wave Dance (Interference)", [
            "Coherent light sources, like lasers, produce synchronized waves.",
            "When waves overlap, they create an interference pattern.",
            "Constructive interference happens when peaks align for greater amplitude.",
            "Destructive interference occurs when waves cancel each other out.",
            "These patterns encode the light's amplitude and phase."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        laser_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        laser1 = SVGMobject(laser_path, color=WHITE).scale(0.4)
        laser2 = SVGMobject(laser_path, color=WHITE).scale(0.4)
        
        # Repositioned from B1/E1 to B2/E2 to avoid clutter (Resolves Issue 28)
        self.place_at_grid(laser1, "B2")
        self.place_at_grid(laser2, "E2")
        
        wave1 = FunctionGraph(lambda x: 0.3 * np.sin(x * 3), x_range=[0, 4], color=BLUE)
        wave2 = FunctionGraph(lambda x: 0.3 * np.sin(x * 3), x_range=[0, 4], color=TEAL)
        
        # Position waves relative to lasers
        wave1.next_to(laser1, RIGHT, buff=0)
        wave2.next_to(laser2, RIGHT, buff=0)
        
        self.play(FadeIn(laser1), FadeIn(laser2))
        self.play(Create(wave1), Create(wave2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        self.play(FadeOut(wave1), FadeOut(wave2), FadeOut(laser1), FadeOut(laser2))
        
        # Setup for pulse interference
        # Using a central point for the collision/overlap
        center_grid = self.grid["C4"]
        t_tracker = ValueTracker(-2.5)
        
        def pulse_y(x, center, amp=0.6):
            return amp * np.exp(-((x - center)**2) / 0.15)

        def get_pulse_points(center, sign=1):
            return [center_grid + np.array([x, sign * pulse_y(x, center), 0]) for x in np.linspace(-2.5, 2.5, 60)]

        pulse1 = VMobject(color=BLUE)
        pulse2 = VMobject(color=TEAL)
        
        pulse1.add_updater(lambda m: m.set_points_as_corners(get_pulse_points(t_tracker.get_value())))
        pulse2.add_updater(lambda m: m.set_points_as_corners(get_pulse_points(-t_tracker.get_value())))

        self.add(pulse1, pulse2)
        self.play(t_tracker.animate.set_value(0), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        def get_sum_points(center):
            return [center_grid + np.array([x, pulse_y(x, center) + pulse_y(x, -center), 0]) for x in np.linspace(-2.5, 2.5, 60)]
        
        constructive_sum = VMobject(color="#00FF00")
        # Snapshot at center=0
        constructive_sum.set_points_as_corners(get_sum_points(0))
        
        self.play(
            FadeOut(pulse1), FadeOut(pulse2),
            FadeIn(constructive_sum)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        
        def get_diff_points(center):
            # One pulse inverted for destructive interference
            return [center_grid + np.array([x, pulse_y(x, center) - pulse_y(x, -center), 0]) for x in np.linspace(-2.5, 2.5, 60)]
        
        destructive_sum = VMobject(color="#FF0000")
        t_tracker.set_value(-2.5)
        
        # Reset pulses for destructive demonstration
        pulse1_d = VMobject(color=BLUE)
        pulse2_d = VMobject(color=TEAL)
        
        pulse1_d.add_updater(lambda m: m.set_points_as_corners(get_pulse_points(t_tracker.get_value())))
        pulse2_d.add_updater(lambda m: m.set_points_as_corners(get_pulse_points(-t_tracker.get_value(), sign=-1)))
        
        self.play(FadeOut(constructive_sum))
        self.add(pulse1_d, pulse2_d)
        
        # Move them to overlap
        self.play(t_tracker.animate.set_value(0), run_time=2, rate_func=linear)
        
        # Show the flattened sum at the moment of overlap
        destructive_sum.set_points_as_corners(get_diff_points(0))
        self.add(destructive_sum)
        self.play(FadeOut(pulse1_d), FadeOut(pulse2_d))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        
        self.play(FadeOut(destructive_sum))
        
        # Phasor representation
        # Moved circle and label system to D5 to better use space (Resolves Issue 29)
        circle = Circle(radius=1.0, color=GREY_A)
        self.place_at_grid(circle, "D5")
        
        phasor_start = self.grid["D5"]
        phasor = Arrow(start=phasor_start, end=phasor_start + RIGHT, color=PURPLE, buff=0)
        
        # Define labels once
        amp_label = Text("Amplitude (A)", font_size=18, color=PURPLE)
        phase_label = Text("Phase (phi)", font_size=18, color=PURPLE)
        
        amp_label.next_to(circle, UP, buff=0.5)
        phase_label.next_to(circle, DOWN, buff=0.5)

        self.play(Create(circle), Write(amp_label), Write(phase_label))
        
        angle_tracker = ValueTracker(0)
        
        def update_phasor(m):
            angle = angle_tracker.get_value()
            m.put_start_and_end_on(
                phasor_start, 
                phasor_start + np.array([np.cos(angle), np.sin(angle), 0])
            )
            
        phasor.add_updater(update_phasor)
        self.add(phasor)
        
        # Rotate phasor to show phase
        self.play(angle_tracker.animate.set_value(TAU * 0.75), run_time=3, rate_func=linear)
        self.wait(2)
