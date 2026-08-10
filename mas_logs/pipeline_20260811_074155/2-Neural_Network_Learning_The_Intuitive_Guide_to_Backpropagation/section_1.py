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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        title_text = "The Big Picture: Learning via Trial and Error"
        lecture_lines = [
            "Neural networks learn by minimizing prediction errors.",
            "Meet Robo-Chef, our robot learning to bake cakes.",
            "It starts with guesses and adjusts through feedback."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Line: "Neural networks learn by minimizing prediction errors."
        self.lecture[0].set_color(YELLOW)
        error_formula = MathTex(r"\text{Error} = |\text{Guess} - \text{Reality}|", font_size=36, color=YELLOW)
        self.place_in_area(error_formula, "A2", "A5")
        self.play(Write(error_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Meet Robo-Chef, our robot learning to bake cakes."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        # Robo-Chef Mobject (Stylized using basic shapes as no assets provided)
        body = Circle(radius=0.4, color="#00FFFF", fill_opacity=0.5)
        eye_l = Dot(radius=0.05, color=WHITE).shift(LEFT*0.1 + UP*0.1)
        eye_r = Dot(radius=0.05, color=WHITE).shift(RIGHT*0.1 + UP*0.1)
        antenna = Line(UP*0.4, UP*0.6, color="#00FFFF")
        robo_chef = VGroup(body, eye_l, eye_r, antenna)
        self.place_at_grid(robo_chef, "C2")
        
        # Cake Mobject (Stylized)
        cake_base = RoundedRectangle(height=0.5, width=0.7, corner_radius=0.1, color="#D2B48C", fill_opacity=1)
        icing = Line(LEFT*0.3, RIGHT*0.3, color=WHITE).shift(UP*0.2)
        cake = VGroup(cake_base, icing)
        self.place_at_grid(cake, "C5")
        
        self.play(FadeIn(robo_chef), FadeIn(cake))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "It starts with guesses and adjusts through feedback."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Slider setup
        slider_line = NumberLine(
            x_range=[0, 60, 10], 
            length=4, 
            include_numbers=True, 
            font_size=18, 
            color=WHITE
        )
        self.place_in_area(slider_line, "E2", "E5")
        
        # Target mark (30 mins)
        target_label = Text("Target: 30", font_size=16, color="#00FF00")
        target_arrow = Arrow(DOWN, UP, color="#00FF00", buff=0.1).scale(0.3)
        target_pos = slider_line.n2p(30)
        target_group = VGroup(target_label, target_arrow).next_to(target_pos, DOWN, buff=0.1)
        
        # Guess handle (Starts at 50 mins)
        guess_val = ValueTracker(50)
        handle = Triangle(color=RED, fill_opacity=1).scale(0.15).rotate(PI)
        handle.add_updater(lambda m: m.move_to(slider_line.n2p(guess_val.get_value()) + UP*0.3))
        
        guess_label = Text("Guess", font_size=16, color=RED)
        guess_label.add_updater(lambda m: m.next_to(handle, UP, buff=0.1))
        
        self.play(Create(slider_line), FadeIn(target_group))
        self.play(FadeIn(handle), FadeIn(guess_label))
        self.wait(1)
        
        # Error Flash and Adjustment
        flash_rect = Rectangle(
            width=6, height=4, 
            fill_color="#330000", 
            fill_opacity=0.6, 
            stroke_width=0
        ).move_to(self.grid["D3"])
        
        self.add_foreground_mobjects(self.title, self.lecture) # Ensure text is visible over flash
        
        self.play(FadeIn(flash_rect))
        self.play(flash_rect.animate.set_fill(opacity=0), run_time=0.5)
        self.play(guess_val.animate.set_value(30), run_time=2)
        
        self.remove(flash_rect)
        self.wait(2)
