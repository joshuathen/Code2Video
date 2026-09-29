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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Applied Visualization: The Orbiting Satellite", 
                          ["Visualize a satellite's elliptical path.", 
                           "Identify the implicit equation for the orbit.", 
                           "Calculate the slope at a specific point."])
        
        # Ellipse: x^2 + 4y^2 = 8 => (x/sqrt(8))^2 + (y/sqrt(2))^2 = 1
        # a = sqrt(8) approx 2.828, b = sqrt(2) approx 1.414
        ellipse = Ellipse(width=2*2.828*0.4, height=2*1.414*0.4, color=WHITE)
        # Positioned per VideoCritic constraints (Issue 39)
        self.place_in_area(ellipse, 'C3', 'E6', scale_factor=0.7)
        
        # Satellite icon (Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg)
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        satellite.set_color("#FF4500")
        satellite.move_to(ellipse.point_from_proportion(0.25))
        
        # Tangent vector
        vector = Arrow(start=ORIGIN, end=RIGHT, color="#00FFFF", buff=0)
        vector.move_to(satellite)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Create(ellipse))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#7FFF00")
        equation = MathTex(r"x^2 + 4y^2 = 8", color=WHITE)
        # Positioned per VideoCritic constraints (Issue 39)
        self.place_at_grid(equation, 'A4', scale_factor=0.9)
        self.play(Write(equation))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeIn(satellite))
        
        # Trace orbit and tangent
        self.play(
            MoveAlongPath(satellite, ellipse),
            UpdateFromFunc(vector, lambda m: m.move_to(satellite)),
            run_time=3
        )
        
        self.play(Flash(satellite))
