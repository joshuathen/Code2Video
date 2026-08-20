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
        self.setup_layout("Summary and Real-world Application", [
            "Dandelin spheres link 3D to 2D.", 
            "They explain planetary orbit mechanics.", 
            "Math governs the physical world."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Summary table of conic section types
        table = Table(
            [["Circle", "e=0"], ["Ellipse", "0<e<1"], ["Parabola", "e=1"], ["Hyperbola", "e>1"]],
            col_labels=[Text("Type"), Text("Eccentricity")],
        )
        self.place_in_area(table, 'A1', 'C3', scale_factor=0.6)
        self.play(Create(table))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Flash focus-directrix property
        fd_text = Text("Focus-Directrix Property", color=YELLOW, font_size=24)
        self.place_at_grid(fd_text, 'D5', scale_factor=0.7)
        self.play(FadeIn(fd_text))
        self.play(Indicate(fd_text, color=YELLOW))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Display an orbit with planet and satellite icons
        orbit = Ellipse(width=2.5, height=1.5, color=GREEN)
        self.place_at_grid(orbit, 'E6', scale_factor=0.9)
        
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg").scale(0.2)
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg").scale(0.15)
        
        # Position planet and satellite on the orbit
        planet.move_to(orbit.get_right())
        satellite.move_to(orbit.get_left())
        
        self.play(Create(orbit), FadeIn(planet), FadeIn(satellite))
        self.play(
            MoveAlongPath(planet, orbit),
            MoveAlongPath(satellite, orbit),
            run_time=3, 
            rate_func=linear
        )
        self.lecture[2].set_color(GREEN)
        
        self.wait(2)
