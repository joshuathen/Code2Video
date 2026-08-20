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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Transitions define how outbreaks evolve.",
            "Susceptible individuals become infected by contact.",
            "Infected individuals eventually transition to recovered."
        ]
        self.setup_layout("Defining the Compartments and Dynamics", lecture_lines)
        
        # Create compartments
        s_rect = Square(side_length=1.5, color=GREEN).set_fill(GREEN, opacity=0.3)
        s_label = Text("S", font_size=36).move_to(s_rect.get_center())
        s_group = VGroup(s_rect, s_label)
        
        i_rect = Square(side_length=1.5, color=RED).set_fill(RED, opacity=0.3)
        i_label = Text("I", font_size=36).move_to(i_rect.get_center())
        i_group = VGroup(i_rect, i_label)
        
        r_rect = Square(side_length=1.5, color=BLUE).set_fill(BLUE, opacity=0.3)
        r_label = Text("R", font_size=36).move_to(r_rect.get_center())
        r_group = VGroup(r_rect, r_label)
        
        # Position them
        self.place_at_grid(s_group, 'B2', scale_factor=0.6)
        self.place_at_grid(i_group, 'B4', scale_factor=0.6)
        self.place_at_grid(r_group, 'B6', scale_factor=0.6)
        
        arrow1 = Arrow(s_group.get_right(), i_group.get_left(), buff=0.1)
        beta_label = MathTex(r"\beta").next_to(arrow1, UP, buff=0.1)
        
        arrow2 = Arrow(i_group.get_right(), r_group.get_left(), buff=0.1)
        gamma_label = MathTex(r"\gamma").next_to(arrow2, UP, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(s_group), FadeIn(i_group), FadeIn(r_group))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(arrow1), Write(beta_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Create(arrow2), Write(gamma_label))
        self.wait(2)
