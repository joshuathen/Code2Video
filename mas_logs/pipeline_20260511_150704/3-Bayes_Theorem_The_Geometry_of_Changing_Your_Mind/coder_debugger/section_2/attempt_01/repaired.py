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
        # Setup layout
        lines = [
            'A camera glints when it detects a squirrel.',
            'It glints for eighty percent of Golden Squirrels.',
            'We shade this area to show true positives.',
            'It also glints for twenty percent of Grey Squirrels.',
            'This shaded region represents our false positives.'
        ]
        self.setup_layout("The Evidence Filter (Likelihood)", lines)

        # Base Strips (Prior representation from previous section)
        gold_base = Rectangle(width=1.8, height=4.0, color="#FFD700", fill_opacity=0.1)
        self.place_in_area(gold_base, "B1", "E3")
        
        grey_base = Rectangle(width=1.8, height=4.0, color="#808080", fill_opacity=0.1)
        self.place_in_area(grey_base, "B4", "E6")

        gold_label_bottom = Text("Golden", font_size=20, color="#FFD700")
        self.place_at_grid(gold_label_bottom, "F2")
        
        grey_label_bottom = Text("Grey", font_size=20, color="#808080")
        self.place_at_grid(grey_label_bottom, "F5")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(
            FadeIn(gold_base),
            FadeIn(grey_base),
            Write(gold_label_bottom),
            Write(grey_label_bottom)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        gold_line_y = gold_base.get_bottom()[1] + (4.0 * 0.8)
        gold_line = Line(
            start=[gold_base.get_left()[0], gold_line_y, 0],
            end=[gold_base.get_right()[0], gold_line_y, 0],
            color=WHITE, stroke_width=2
        )
        
        self.play(Create(gold_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Shade the top 80% (Likelihood region)
        gold_shade = Rectangle(
            width=1.8, height=3.2, 
            color="#FFFACD", fill_opacity=0.6, stroke_width=0
        )
        gold_shade.move_to(gold_base.get_top() + DOWN * (3.2 / 2))
        
        # Prob Label - Using Text instead of MathTex to avoid LaTeX dependency
        gold_prob_label = Text("P(Glint|Gold) = 0.8", font_size=18, color="#FFFACD")
        self.place_at_grid(gold_prob_label, "A2")
        
        self.play(
            FadeIn(gold_shade),
            Write(gold_prob_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # 20% line for Grey Strip
        grey_line_y = grey_base.get_bottom()[1] + (4.0 * 0.2)
        grey_line = Line(
            start=[grey_base.get_left()[0], grey_line_y, 0],
            end=[grey_base.get_right()[0], grey_line_y, 0],
            color=WHITE, stroke_width=2
        )
        
        self.play(Create(grey_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Shade top 20% of Grey
        grey_shade = Rectangle(
            width=1.8, height=0.8, 
            color="#D3D3D3", fill_opacity=0.6, stroke_width=0
        )
        grey_shade.move_to(grey_base.get_top() + DOWN * (0.8 / 2))
        
        # Prob Label - Using Text instead of MathTex to avoid LaTeX dependency
        grey_prob_label = Text("P(Glint|Grey) = 0.2", font_size=18, color="#D3D3D3")
        self.place_at_grid(grey_prob_label, "A5")
        
        self.play(
            FadeIn(grey_shade),
            Write(grey_prob_label)
        )
        self.wait(2)
        self.lecture[4].set_color(WHITE)