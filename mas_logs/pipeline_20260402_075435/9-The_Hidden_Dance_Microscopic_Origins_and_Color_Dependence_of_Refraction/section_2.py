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
        # Setup the layout with the lecture lines
        lecture_lines = [
            "At the microscopic level, atoms consist of charged particles.",
            "Electrons behave like tiny masses on springs.",
            "Light's electric field drives these electrons to vibrate."
        ]
        self.setup_layout("Prerequisite: The Atomic Oscillator Model", lecture_lines)

        # Pre-load Assets
        spring_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/spring.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Nucleus: Red circle
        nucleus = Circle(radius=0.3, color="#FF0000", fill_opacity=1)
        self.place_at_grid(nucleus, "D4")
        nucleus_label = Text("Nucleus", font_size=18, color="#FF0000")
        self.place_at_grid(nucleus_label, "E4", scale_factor=0.8) # Issue 33
        
        # Electron: Blue dot
        electron = Dot(color="#00BFFF", radius=0.15)
        self.place_at_grid(electron, "B4")
        electron_label = Text("Electron", font_size=18, color="#00BFFF")
        self.place_at_grid(electron_label, "A4", scale_factor=0.8) # Issue 32
        
        # Orbit Path: Circular dashed line centered at nucleus (D4) passing through electron (B4)
        radius_val = np.linalg.norm(self.grid["B4"] - self.grid["D4"])
        orbit_path = DashedVMobject(Circle(radius=radius_val, color=WHITE))
        self.place_at_grid(orbit_path, "D4") 
        
        self.play(FadeIn(nucleus), FadeIn(nucleus_label))
        self.play(Create(orbit_path))
        self.play(FadeIn(electron), FadeIn(electron_label))
        
        # Rotate electron along the path briefly
        self.play(Rotate(electron, angle=PI, about_point=self.grid["D4"]), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Transition: Remove orbit and label, reposition electron to rest state at B4
        self.play(FadeOut(orbit_path), FadeOut(electron_label))
        self.play(electron.animate.move_to(self.grid["B4"]))
        
        # Load and place spring SVG asset connecting nucleus and electron
        spring = SVGMobject(spring_asset_path, color="#AAAAAA")
        spring_start = self.grid["D4"]
        spring_end = self.grid["B4"]
        
        initial_height = np.linalg.norm(spring_end - spring_start)
        spring.stretch_to_fit_height(initial_height)
        spring.move_to((spring_start + spring_end) / 2)
        
        self.play(FadeIn(spring))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Change electron color to grey as described
        self.play(electron.animate.set_color("#888888"))
        
        # Add yellow E-field arrow and label
        e_field_label = Text("E-field", font_size=20, color="#FFFF00")
        self.place_at_grid(e_field_label, "A3", scale_factor=0.8) # Issue 31
        
        # Create the oscillating arrow centered at C3
        e_field_arrow = Arrow(
            start=self.grid["D3"], 
            end=self.grid["B3"], 
            color="#FFFF00", 
            buff=0,
            stroke_width=6
        )
        
        # Oscillating parameters
        time_tracker = ValueTracker(0)
        amplitude_val = 0.6
        omega = 2 * PI
        
        nucleus_pos = self.grid["D4"]
        electron_rest_pos = self.grid["B4"]
        arrow_center = self.grid["C3"]
        
        def update_electron(m):
            t = time_tracker.get_value()
            shift = UP * amplitude_val * np.sin(omega * t)
            m.move_to(electron_rest_pos + shift)
            
        def update_spring(m):
            # Stretch spring between nucleus and current electron position
            e_pos = electron.get_center()
            current_height = np.linalg.norm(e_pos - nucleus_pos)
            if current_height > 0.01:
                m.stretch_to_fit_height(current_height)
                m.move_to((nucleus_pos + e_pos) / 2)
            
        def update_arrow(m):
            t = time_tracker.get_value()
            scale_factor = np.sin(omega * t)
            # Arrow oscillates in size and direction relative to its center at C3
            m.put_start_and_end_on(
                arrow_center - UP * scale_factor * 0.4, 
                arrow_center + UP * scale_factor * 0.4
            )

        electron.add_updater(update_electron)
        spring.add_updater(update_spring)
        e_field_arrow.add_updater(update_arrow)
        
        self.add(e_field_arrow, e_field_label)
        self.play(time_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        
        # Stop updaters
        electron.remove_updater(update_electron)
        spring.remove_updater(update_spring)
        e_field_arrow.remove_updater(update_arrow)
        
        self.wait(2)
