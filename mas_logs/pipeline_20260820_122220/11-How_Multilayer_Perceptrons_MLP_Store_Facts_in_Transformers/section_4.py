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
        self.setup_layout("Integration: Why MLPs vs. Attention?", [
            "Attention routes information between tokens.",
            "MLPs store long-term factual memories.",
            "Attention coordinates; MLPs hold the knowledge."
        ])
        
        # Load assets
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        filing = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filing.svg")
        library = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")

        # === Animation for Lecture Line 1 ===
        # Show attention mechanism (#FF5733) connecting different token nodes
        self.lecture[0].set_color("#FF5733")
        nodes = VGroup(*[brain.copy().scale(0.5) for _ in range(3)])
        self.place_at_grid(nodes[0], 'B4')
        self.place_at_grid(nodes[1], 'B6')
        self.place_at_grid(nodes[2], 'C5')
        lines = VGroup(Line(nodes[0].get_center(), nodes[1].get_center(), color="#FF5733"),
                       Line(nodes[1].get_center(), nodes[2].get_center(), color="#FF5733"),
                       Line(nodes[0].get_center(), nodes[2].get_center(), color="#FF5733"))
        self.play(Create(nodes), Create(lines))

        # === Animation for Lecture Line 2 ===
        # Show MLP storage (#33FF57) as a dense block of memory
        self.lecture[1].set_color("#33FF57")
        mem_block = filing.copy().scale(1.2)
        self.place_at_grid(mem_block, 'E5')
        self.play(FadeIn(mem_block, shift=UP))

        # === Animation for Lecture Line 3 ===
        # Animate attention routing signals (#FFFFFF) vs MLP memory activation (#FFFF33)
        self.lecture[2].set_color("#FFFFFF")
        signal = Dot(color="#FFFFFF").move_to(nodes[0].get_center())
        activation = library.copy().scale(0.8)
        self.place_at_grid(activation, 'E2')
        
        self.play(signal.animate.move_to(nodes[2].get_center()), run_time=1.5)
        self.play(FadeIn(activation), mem_block.animate.set_color("#FFFF33"))
        self.wait(2)
