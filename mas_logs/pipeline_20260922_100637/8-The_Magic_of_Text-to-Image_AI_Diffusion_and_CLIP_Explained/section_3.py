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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Navigator: CLIP", [
            "CLIP matches text and images together perfectly.",
            "It calculates similarity scores for alignment.",
            "This guides the model towards the target."
        ])
        
        # Elements
        img_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        txt_label = Text("Text: 'Corgi'", font_size=20)
        
        connection = Line(start=ORIGIN, end=RIGHT*2, color="#33A1FF")
        score_meter = Rectangle(width=3, height=0.5, color=GREY)
        score_label = Text("Similarity Score", font_size=16)

        # Placement
        self.place_at_grid(img_icon, 'B2', scale_factor=0.5)
        self.place_at_grid(camera_icon, 'B3', scale_factor=0.5)
        self.place_at_grid(txt_label, 'B5', scale_factor=1.0)
        self.place_at_grid(connection, 'B3', scale_factor=0.9)
        self.place_in_area(score_meter, 'D2', 'D4', scale_factor=0.6)
        self.place_at_grid(score_label, 'D5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(
            self.lecture[0].animate.set_color("#33A1FF"), 
            FadeIn(img_icon), 
            Write(txt_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[1].animate.set_color("#33A1FF"), 
            Create(connection),
            FadeIn(camera_icon),
            Create(score_meter), 
            Write(score_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        score_val = Rectangle(width=2.5, height=0.4, color="#33A1FF", fill_opacity=0.5).move_to(score_meter.get_center())
        self.play(self.lecture[2].animate.set_color("#33A1FF"), FadeIn(score_val))
        self.wait(2)
