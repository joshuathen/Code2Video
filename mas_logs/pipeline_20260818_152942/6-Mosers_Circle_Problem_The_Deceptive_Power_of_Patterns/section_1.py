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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Place n points on a circle's edge.",
            "Connect every pair with a chord.",
            "How many regions are formed inside?",
            "The sequence: 1, 2, 4, 8, 16.",
            "What comes after 16?"
        ]
        self.setup_layout("Introduction: The Curiosity of Connection", lecture_lines)
        
        # Assets / Objects
        # Using Circle primitive as a fallback since SVGMobject might not contain valid paths
        circle = Circle(color=WHITE)
        self.place_at_grid(circle, 'C5', scale_factor=0.6) # Adjusted per issue 18/33
        
        # Dots
        dots = VGroup(*[Dot(circle.point_from_proportion(i/6), color="#00CED1") for i in range(6)])
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeToColor(self.lecture[0], color="#00CED1"))
        self.play(FadeIn(circle), FadeIn(dots))

        # === Animation for Lecture Line 2 ===
        self.play(FadeToColor(self.lecture[1], color="#FFFF00"))
        chords = VGroup(*[Line(dots[i].get_center(), dots[j].get_center(), color="#FFFF00") 
                          for i in range(6) for j in range(i+1, 6)])
        self.play(Create(chords))

        # === Animation for Lecture Line 3 ===
        self.play(FadeToColor(self.lecture[2], color="#ADFF2F"))
        # Regions highlight (simple proxy)
        region = Circle(radius=1.0, color="#ADFF2F", fill_opacity=0.3)
        self.place_at_grid(region, 'C5', scale_factor=0.6)
        self.play(FadeIn(region))

        # === Animation for Lecture Line 4 ===
        self.play(FadeToColor(self.lecture[3], color="#FFFFFF"))
        seq_label = Text("1, 2, 4, 8, 16", font_size=24, color="#FFFFFF")
        self.place_in_area(seq_label, 'D3', 'E6', scale_factor=0.5) # Per issue 19/34
        self.play(Write(seq_label))

        # === Animation for Lecture Line 5 ===
        self.play(FadeToColor(self.lecture[4], color="#FF4500"))
        question = Text("?", font_size=40, color="#FF4500")
        self.place_at_grid(question, 'F5', scale_factor=1.0)
        self.play(FadeIn(question))
        self.wait(1)
