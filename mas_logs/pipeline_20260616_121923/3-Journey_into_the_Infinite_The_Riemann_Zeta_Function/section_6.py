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
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "Application: The Rhythm of Physics", 
            [
                "Zeta isn't just abstract; it's in our universe.", 
                "It tames infinite energy in the Casimir effect.", 
                "This bridge links pure math to physical reality."
            ]
        )
        
        # Color definitions
        PLATE_COLOR = "#C0C0C0"
        WAVE_COLOR = "#00BFFF"
        FORMULA_COLOR = "#ADFF2F"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Metal Plates
        plate_left = Rectangle(width=0.2, height=3.5, color=PLATE_COLOR, fill_opacity=1)
        plate_right = Rectangle(width=0.2, height=3.5, color=PLATE_COLOR, fill_opacity=1)
        
        # Initial positioning (Fixed per Issue 49 and 50)
        self.place_in_area(plate_left, 'C2', 'F2', scale_factor=1.0)
        self.place_in_area(plate_right, 'C5', 'F5', scale_factor=1.0)
        
        self.play(
            FadeIn(plate_left, shift=RIGHT),
            FadeIn(plate_right, shift=LEFT),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Create quantum fluctuations (Blue wavy lines)
        time_tracker = ValueTracker(0)
        
        def create_fluctuation(y_offset):
            # Using ParametricFunction for performance-friendly updates
            # We reference the current positions of the plates to define bounds
            return always_redraw(lambda: 
                ParametricFunction(
                    lambda u: np.array([
                        u, 
                        plate_left.get_center()[1] + y_offset + 0.15 * np.sin(8 * (u - plate_left.get_center()[0]) - 5 * time_tracker.get_value()), 
                        0
                    ]),
                    t_range=[plate_left.get_center()[0] + 0.15, plate_right.get_center()[0] - 0.15],
                    color=WAVE_COLOR,
                    stroke_width=2
                )
            )

        waves = VGroup(
            create_fluctuation(0.8),
            create_fluctuation(0),
            create_fluctuation(-0.8)
        )
        
        self.play(Create(waves))
        # Constant oscillation
        self.add_updater(lambda dt: time_tracker.increment_value(dt))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Calculation formula - (Fixed per Issue 48)
        zeta_formula = Text("ζ(-3)", color=FORMULA_COLOR)
        self.place_in_area(zeta_formula, 'B3', 'B4', scale_factor=1.0)
        
        # Target position for closer plates (Aligned with Issue 49/50 logic)
        # We calculate the center of the target area 'C3' to 'F3'
        target_tl = self.grid['C3']
        target_br = self.grid['F3']
        target_center = (target_tl + target_br) / 2
        
        self.play(
            plate_right.animate.move_to(target_center),
            Write(zeta_formula),
            run_time=2
        )
        
        self.wait(3)
        
        # Final cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
