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
        self.setup_layout("Coordinate Representation & Components", 
                          ["We use grids to map space.", "Vectors break into horizontal components.", "They also have vertical components."])
        
        # Load Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vector_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")

        # === Animation for Lecture Line 1 ===
        # Draw grid lines (#808080)
        grid_asset.set_color("#808080")
        self.place_in_area(grid_asset, 'C3', 'F6', scale_factor=0.5)
        self.play(Create(grid_asset))
        self.lecture[0].set_color("#808080")
        
        # === Animation for Lecture Line 2 ===
        # Draw vector 'v'
        vector_asset.set_color(YELLOW)
        # Assuming grid_asset origin is roughly at grid_asset.get_left()
        vector_asset.move_to(grid_asset.get_corner(DL))
        self.play(FadeIn(vector_asset))
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        horizontal_label = MathTex("3", color="#00FF00")
        self.place_at_grid(horizontal_label, 'C4', scale_factor=0.6)
        
        # Apply title repositioning
        self.title.move_to(self.grid['A3'])
        
        # Drawing components
        x_comp = DashedLine(grid_asset.get_corner(DL), grid_asset.get_corner(DR), color="#00FF00")
        y_comp = DashedLine(grid_asset.get_corner(DR), grid_asset.get_corner(UR), color="#00FF00")
        
        self.play(Create(x_comp), Create(y_comp), Write(horizontal_label))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
