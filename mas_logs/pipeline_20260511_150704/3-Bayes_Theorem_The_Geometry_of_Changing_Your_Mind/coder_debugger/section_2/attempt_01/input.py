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
        # Setup layout with specified content
        lecture_lines = [
            "A camera glints when detecting a squirrel's presence.",
            "We shade areas where the camera triggers a glint.",
            "These heights represent the likelihood of observing a glint."
        ]
        self.setup_layout("The Evidence Filter (Likelihood)", lecture_lines)

        # Pre-create base strips
        # Using dimensions that fit in B1-E3 and B4-E6 areas
        gold_base = Rectangle(width=2.2, height=4.0, color="#FFD700", fill_opacity=0.1)
        self.place_in_area(gold_base, "B1", "E3")
        
        grey_base = Rectangle(width=2.2, height=4.0, color="#808080", fill_opacity=0.1)
        self.place_in_area(grey_base, "B4", "E6")

        # Bottom Labels - Issues 26 & 27 fix (scaling for safety)
        gold_label_bottom = Text("Golden", font_size=20, color="#FFD700")
        self.place_at_grid(gold_label_bottom, "F2", scale_factor=0.7)
        
        grey_label_bottom = Text("Grey", font_size=20, color="#808080")
        self.place_at_grid(grey_label_bottom, "F5", scale_factor=0.7)

        # Asset: camera icon - Issue 21
        camera_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/camera.svg")
        self.place_at_grid(camera_icon, "A3", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Horizontal lines: 80% gold, 20% grey
        gold_line_y = gold_base.get_bottom()[1] + (gold_base.height * 0.8)
        gold_line = Line(
            start=[gold_base.get_left()[0], gold_line_y, 0],
            end=[gold_base.get_right()[0], gold_line_y, 0],
            color=WHITE, stroke_width=2
        )
        
        grey_line_y = grey_base.get_bottom()[1] + (grey_base.height * 0.2)
        grey_line = Line(
            start=[grey_base.get_left()[0], grey_line_y, 0],
            end=[grey_base.get_right()[0], grey_line_y, 0],
            color=WHITE, stroke_width=2
        )

        self.play(
            FadeIn(gold_base), FadeIn(grey_base),
            Write(gold_label_bottom), Write(grey_label_bottom),
            run_time=1
        )
        self.play(Create(gold_line), Create(grey_line), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Shade areas: bright yellow (#FFFACD) and light grey (#D3D3D3)
        gold_shade = Rectangle(
            width=gold_base.width, height=gold_base.height * 0.8,
            color="#FFFACD", fill_opacity=0.6, stroke_width=0
        )
        gold_shade.move_to(gold_base.get_top() + DOWN * (gold_shade.height / 2))
        
        grey_shade = Rectangle(
            width=grey_base.width, height=grey_base.height * 0.2,
            color="#D3D3D3", fill_opacity=0.6, stroke_width=0
        )
        grey_shade.move_to(grey_base.get_top() + DOWN * (grey_shade.height / 2))

        self.play(FadeIn(gold_shade), FadeIn(grey_shade))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Probability labels - Issues 21 & 25
        gold_prob_label = Text("P(Glint|Gold)=0.8", font_size=18, color="#FFFACD")
        self.place_at_grid(gold_prob_label, "A2", scale_factor=0.8)
        
        grey_prob_label = Text("P(Glint|Grey)=0.2", font_size=18, color="#D3D3D3")
        self.place_at_grid(grey_prob_label, "A5", scale_factor=0.6)

        self.play(
            FadeIn(camera_icon),
            Write(gold_prob_label),
            Write(grey_prob_label)
        )
        self.wait(2)
        self.lecture[2].set_color(WHITE)
