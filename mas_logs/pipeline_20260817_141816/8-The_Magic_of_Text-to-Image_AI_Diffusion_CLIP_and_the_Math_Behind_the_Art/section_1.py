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
        self.setup_layout("Introduction: The Creative Loop", [
            "AI converts text prompts into visual art.",
            "CLIP acts as our understanding engine.",
            "Diffusion serves as our creation engine."
        ])
        
        # Visual assets
        node_imgs = [
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        ]
        
        for img in node_imgs:
            img.set_color("#FFD700")
            
        labels = VGroup(*[Text(t, font_size=16) for t in ["Prompt", "CLIP", "Diffusion"]])
        
        # Positions
        self.place_at_grid(node_imgs[0], 'B4', scale_factor=0.8) # Prompt
        self.place_at_grid(node_imgs[1], 'D3', scale_factor=0.8) # CLIP
        self.place_at_grid(node_imgs[2], 'D5', scale_factor=0.8) # Diffusion
        
        self.place_at_grid(labels[0], 'B4', scale_factor=0.8).shift(DOWN * 0.7)
        self.place_at_grid(labels[1], 'D3', scale_factor=0.8).shift(DOWN * 0.7)
        self.place_at_grid(labels[2], 'D5', scale_factor=0.8).shift(DOWN * 0.7)
        
        connections = VGroup(
            Line(node_imgs[0].get_center(), node_imgs[1].get_center(), color=WHITE),
            Line(node_imgs[1].get_center(), node_imgs[2].get_center(), color=WHITE),
            Line(node_imgs[2].get_center(), node_imgs[0].get_center(), color=WHITE)
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(node_imgs[0]), FadeIn(labels[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(FadeIn(node_imgs[1]), FadeIn(labels[1]), Create(connections[0]))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(FadeIn(node_imgs[2]), FadeIn(labels[2]), Create(connections[1]), Create(connections[2]))
        
        self.wait(2)
