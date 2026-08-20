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
        lecture_lines = [
            "CLIP aligns text and image vectors.", 
            "Similar items point the same way.", 
            "It measures proximity using cosine similarity.", 
            "This bridge maps language to vision.", 
            "AI interprets your prompt through CLIP."
        ]
        self.setup_layout("CLIP: The Bridge Between Text and Vision", lecture_lines)
        
        # Objects
        # Using SVGAssets as requested
        text_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg", color="#FFFFFF")
        image_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg", color="#00FFFF")
        vec_t = Dot(color="#FFFFFF")
        vec_i = Dot(color="#00FFFF")
        dist_line = Line(start=vec_t.get_center(), end=vec_i.get_center(), color="#FFFF00")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#FFFF00")
        
        sim_matrix = Rectangle(width=2, height=2, color="#ADFF2F", fill_opacity=0.3)
        concept_match = VGroup(Dot(color="#FF00FF"), Dot(color="#FF00FF")).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(text_obj, 'C2', scale_factor=0.8)
        self.place_at_grid(image_obj, 'C5', scale_factor=0.8)
        self.play(FadeIn(text_obj), FadeIn(image_obj))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_at_grid(vec_t, 'D2', scale_factor=0.5)
        self.place_at_grid(vec_i, 'D5', scale_factor=0.5)
        self.play(FadeIn(vec_t), FadeIn(vec_i))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        dist_line.add_updater(lambda m: m.put_start_and_end_on(vec_t.get_center(), vec_i.get_center()))
        self.place_at_grid(camera_icon, 'D3', scale_factor=0.5)
        self.play(Create(dist_line), FadeIn(camera_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADFF2F"))
        self.place_in_area(sim_matrix, 'E3', 'F5', scale_factor=0.6)
        self.play(FadeIn(sim_matrix))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.place_at_grid(concept_match, 'B3', scale_factor=0.7)
        self.play(FadeIn(concept_match))
        self.wait(1)
