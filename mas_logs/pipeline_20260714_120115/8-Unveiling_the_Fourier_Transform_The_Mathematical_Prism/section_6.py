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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define lecture lines for Section 6
        lecture_lines = [
            "Time and frequency domains show the same information differently.",
            "The Fourier Transform is the bridge between these views.",
            "It reveals the simple structure within our complex world."
        ]
        
        # Initialize layout
        self.setup_layout("Summary: One World, Two Views", lecture_lines)
        
        # Visual styling
        TIME_COLOR = BLUE_C
        FREQ_COLOR = ORANGE
        BRIDGE_COLOR = "#FFD700"  # Gold as requested

        # === Animation for Lecture Line 1 ===
        # L1: Time and frequency domains show the same information differently.
        # Step: Show "Time" (left) and "Frequency" (right) labels.
        self.lecture[0].set_color(WHITE)
        
        time_label = Text("Time Domain", font_size=20, color=WHITE)
        freq_label = Text("Frequency Domain", font_size=20, color=WHITE)
        
        # Position labels at B2 and B5 (Fix Issue 42 & 43: scale_factor=0.6)
        self.place_at_grid(time_label, 'B2', scale_factor=0.6)
        self.place_at_grid(freq_label, 'B5', scale_factor=0.6)
        
        self.play(
            Write(time_label),
            Write(freq_label),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: The Fourier Transform is the bridge between these views.
        # Step: Display a beating heart wave and a single spike.
        self.lecture[1].set_color(WHITE)
        
        # Create Heart Wave (Time Domain)
        ecg_pts = [
            [-0.5, 0, 0], [-0.3, 0, 0], [-0.25, 0.1, 0], [-0.2, 0, 0],
            [-0.15, -0.1, 0], [-0.1, 0.6, 0], [-0.05, -0.2, 0], [0, 0, 0],
            [0.1, 0.2, 0], [0.2, 0, 0], [0.5, 0, 0]
        ]
        heart_wave = VMobject(color=TIME_COLOR)
        heart_wave.set_points_as_corners([np.array(p) for p in ecg_pts])
        
        # Create Single Spike (Frequency Domain)
        spike_pts = [
            [-0.5, 0, 0], [-0.05, 0, 0], [0, 0.8, 0], [0.05, 0, 0], [0.5, 0, 0]
        ]
        spike = VMobject(color=FREQ_COLOR)
        spike.set_points_as_corners([np.array(p) for p in spike_pts])
        
        # Position graphs in Row C (1 grid unit below Row B labels)
        self.place_at_grid(heart_wave, 'C2', scale_factor=1.2)
        self.place_at_grid(spike, 'C5', scale_factor=1.2)
        
        self.play(
            Create(heart_wave),
            Create(spike),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L3: It reveals the simple structure within our complex world.
        # Step: Draw a gold #FFD700 bridge arrow between the views.
        self.lecture[2].set_color(BRIDGE_COLOR)
        
        # Bridge Arrow connecting the two domains
        # Spanning C3 to C4
        bridge_arrow = DoubleArrow(LEFT, RIGHT, color=BRIDGE_COLOR, stroke_width=4, buff=0.1)
        self.place_in_area(bridge_arrow, 'C3', 'C4', scale_factor=1.0)
        
        # Bridge Label (Fix Issue 41: Move to A3-A4 and scale to 0.7)
        ft_text = Text("Fourier Transform", font_size=18, color=BRIDGE_COLOR)
        self.place_in_area(ft_text, 'A3', 'A4', scale_factor=0.7)
        
        self.play(
            GrowFromCenter(bridge_arrow),
            FadeIn(ft_text),
            run_time=1.5
        )
        self.wait(3)
