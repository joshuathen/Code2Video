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
        # Setup title and lecture lines
        self.setup_layout(
            "The Microscopic Origin: The Lorentz Oscillator Model", 
            [
                'Matter consists of electrons bound to atomic nuclei.', 
                'Model these interactions as electrons on tiny springs.', 
                'Incoming electromagnetic waves force these electrons to oscillate.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Create Nucleus (Red) and Electron (Blue)
        nucleus = Circle(radius=0.35, color="#FF0000", fill_opacity=1.0)
        self.place_at_grid(nucleus, 'D3', scale_factor=1.0)
        
        electron = Dot(color="#0000FF", radius=0.15)
        self.place_at_grid(electron, 'B3', scale_factor=1.0)
        
        # Labels for the components
        nucleus_label = Text("Nucleus", font_size=16, color="#FF0000")
        self.place_at_grid(nucleus_label, 'D4', scale_factor=1.0)
        
        electron_label = Text("Electron", font_size=16, color="#0000FF")
        self.place_at_grid(electron_label, 'B4', scale_factor=1.0)
        
        self.play(FadeIn(nucleus), FadeIn(nucleus_label))
        self.play(FadeIn(electron), FadeIn(electron_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlighting
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create a zigzag spring (White)
        spring = VMobject(color="#FFFFFF")
        
        def update_spring(mob):
            start_pos = nucleus.get_center()
            end_pos = electron.get_center()
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            if length < 0.1:
                mob.set_points_as_corners([start_pos, end_pos])
                return
            unit_dir = direction / length
            normal = np.array([-unit_dir[1], unit_dir[0], 0])
            
            num_points = 14
            pts = [start_pos]
            for i in range(1, num_points):
                alpha = i / num_points
                # Alternating zigzag offsets
                side_offset = 0.25 if i % 2 == 1 else -0.25
                pts.append(start_pos + alpha * direction + side_offset * normal)
            pts.append(end_pos)
            mob.set_points_as_corners(pts)
            
        # Initialize and show the spring
        update_spring(spring)
        self.play(Create(spring))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlighting
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Tracker for time/wave phase
        time_tracker = ValueTracker(0)
        
        # Oscillating wave passing through the system (Light Blue)
        wave = VMobject(color="#00FFFF")
        def update_wave(mob):
            t = time_tracker.get_value()
            pts = []
            # Wave centered at B-row y-level (1.2) where the electron is
            for x_val in np.linspace(0.5, 6.0, 50):
                # Harmonic wave equation
                y_val = 1.2 + 0.6 * np.sin(2 * PI * (x_val - 0.5) / 2.0 - 5 * t)
                pts.append([x_val, y_val, 0])
            mob.set_points_as_corners(pts)
            
        wave.add_updater(update_wave)
        
        # Electron oscillates based on wave phase at its fixed x-position (x=2.5)
        # At x=2.5, phase is 2*PI*(2.5-0.5)/2.0 - 5*t = 2*PI - 5*t
        electron.add_updater(lambda m: m.move_to([
            2.5, 
            1.2 + 0.6 * np.sin(2 * PI - 5 * time_tracker.get_value()), 
            0
        ]))
        
        # Spring and Label follow the electron's movement
        spring.add_updater(update_spring)
        electron_label.add_updater(lambda m: m.next_to(electron, RIGHT, buff=0.2))
        
        self.add(wave)
        # Animate wave passing and causing oscillation
        self.play(time_tracker.animate.set_value(6), run_time=6, rate_func=linear)
        
        # Finish highlighting
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
