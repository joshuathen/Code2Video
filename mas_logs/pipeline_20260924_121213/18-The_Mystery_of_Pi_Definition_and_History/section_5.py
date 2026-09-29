from manim import *

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
        self.setup_layout("Summary & Real-World Application", [
            "Pi links lines and circles.",
            "[Asset: gps_satellite_orbit]",
            "GPS uses Pi for navigation.",
            "[Asset: planetary_surface_path]",
            "It is essential for modern space travel."
        ])
        
        # Path to satellite asset
        satellite_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg"

        # === Animation for Lecture Line 1 ===
        # Pi links lines and circles.
        circle = Circle(radius=0.7, color="#00BFFF")
        line = Line(start=LEFT*1, end=RIGHT*1, color="#00BFFF")
        self.place_at_grid(circle, 'B2', scale_factor=0.6)
        self.place_at_grid(line, 'B5', scale_factor=0.6)
        self.play(Create(circle), Create(line))
        self.lecture[0].set_color("#00BFFF")

        # === Animation for Lecture Line 2 ===
        # [Asset: gps_satellite_orbit]
        # Use SVG asset
        orbit = Ellipse(width=3, height=1.5, color="#D3D3D3")
        satellite = SVGMobject(satellite_path, color=YELLOW)
        self.place_at_grid(orbit, 'E3', scale_factor=0.6)
        self.place_at_grid(satellite, 'E3', scale_factor=0.4)
        
        satellite.add_updater(lambda d, dt: d.move_to(orbit.point_from_proportion((self.time * 0.2) % 1)))
        self.play(Create(orbit), FadeIn(satellite))
        self.lecture[1].set_color("#D3D3D3")

        # === Animation for Lecture Line 3 ===
        # GPS uses Pi for navigation.
        text_gps = Text("GPS uses Pi", font_size=24, color="#FFFFFF")
        self.place_at_grid(text_gps, 'C4', scale_factor=0.75)
        self.play(Write(text_gps))
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        # [Asset: planetary_surface_path]
        planet = Circle(radius=0.5, color="#00FF00")
        path = Arc(radius=0.5, start_angle=0, angle=PI/2, color="#00FF00")
        self.place_at_grid(planet, 'E5', scale_factor=0.6)
        self.place_at_grid(path, 'E6', scale_factor=0.7)
        self.play(Create(planet), Create(path))
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        # It is essential for modern space travel.
        text_essential = Text("Essential Math", font_size=24, color="#FF4500")
        self.place_in_area(text_essential, 'A3', 'A5', scale_factor=0.6)
        self.play(Write(text_essential))
        self.lecture[4].set_color("#FF4500")
        
        self.wait(2)
