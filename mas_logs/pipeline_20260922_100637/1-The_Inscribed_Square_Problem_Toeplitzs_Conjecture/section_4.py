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
        lecture_lines = [
            "Visualize a square template sliding along.",
            "Rotate and shrink the square carefully.",
            "All four corners eventually touch.",
            "The intersection points click perfectly.",
            "A square is now fully inscribed."
        ]
        self.setup_layout("Visualizing the Inscription", lecture_lines)
        
        # Elements
        curve = ParametricFunction(
            lambda t: np.array([2*np.cos(t) + 0.5*np.sin(3*t), 1.5*np.sin(t), 0]),
            t_range=[0, 2*PI]
        ).set_color(BLUE)
        
        # Asset Loading (placeholder for image import logic)
        # Using SVGMobject for template.svg as it is a common way to load SVG assets in Manim
        try:
            template = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/template.svg")
        except:
            template = Square() # Fallback

        self.place_in_area(curve, 'C3', 'F6', scale_factor=0.8) # Adjusted per VideoCritic
        
        square = Square(side_length=1.5).set_color(WHITE)
        self.place_at_grid(square, 'D4', scale_factor=0.6) # Adjusted per VideoCritic
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"), FadeIn(curve), FadeIn(template), FadeIn(square))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"), 
                  Rotate(square, angle=PI/6),
                  square.animate.scale(0.7))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"),
                  square.animate.move_to(self.grid['D4']).scale(0.8))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"),
                  square.animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#32CD32"),
                  Flash(square, color="#32CD32"))
