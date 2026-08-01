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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialization
        lines = [
            'In 3D, the determinant scales the volume of boxes.', 
            'It remains the universal expansion factor for any dimension.', 
            'Think of the determinant as the magnitude of stretching.'
        ]
        self.setup_layout("Scaling Up: 3D and Summary", lines)
        
        # Asset Path
        box_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/box.svg"

        # Colors
        color_l1 = "#87CEEB" # Sky Blue
        color_l2 = "#FFD700" # Gold
        color_l3 = "#98FB98" # Pale Green
        color_det = "#FFD700"
        color_flash = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_l1))

        # Create a wireframe cube in perspective
        # Perspective shift
        ps = np.array([0.3, 0.3, 0])
        
        # Square vertices
        v_front = [np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0]), np.array([0,1,0])]
        v_back = [v + ps for v in v_front]
        
        front_face = Polygon(*v_front, color=color_l1, stroke_width=2)
        back_face = Polygon(*v_back, color=color_l1, stroke_width=2, stroke_opacity=0.5)
        connectors = VGroup(*[Line(v_front[i], v_back[i], color=color_l1, stroke_width=2, stroke_opacity=0.7) for i in range(4)])
        
        wireframe_cube = VGroup(front_face, back_face, connectors)
        self.place_in_area(wireframe_cube, "C2", "E4", scale_factor=1.5)
        
        self.play(Create(wireframe_cube))
        self.wait(1)

        # Skewed box asset
        box_svg = SVGMobject(box_asset_path)
        box_svg.set_color(color_l1)
        self.place_in_area(box_svg, "C2", "E4", scale_factor=1.8)

        self.play(ReplacementTransform(wireframe_cube, box_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_l2))

        # "Determinant" summary
        det_text = Text("Determinant", color=color_det, font_size=24)
        self.place_at_grid(det_text, "A3", scale_factor=1.2)
        
        area_label = Text("2D: Area", color=WHITE, font_size=20)
        vol_label = Text("3D: Volume", color=WHITE, font_size=20)
        self.place_at_grid(area_label, "B2", scale_factor=0.8)
        self.place_at_grid(vol_label, "B4", scale_factor=0.8)
        
        link1 = Line(det_text.get_bottom(), area_label.get_top(), color=color_l2, buff=0.1)
        link2 = Line(det_text.get_bottom(), vol_label.get_top(), color=color_l2, buff=0.1)

        self.play(
            Write(det_text),
            run_time=1
        )
        self.play(
            Create(link1),
            Create(link2),
            FadeIn(area_label),
            FadeIn(vol_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_l3))

        # Volume value label
        vol_formula = Text("Volume = det(A)", color=WHITE, font_size=24)
        self.place_at_grid(vol_formula, "F3", scale_factor=1.0)

        # Flash and Show Formula
        self.play(
            box_svg.animate.set_color(color_flash),
            Flash(box_svg, color=color_flash, flash_radius=1.2),
            Write(vol_formula),
            run_time=1.5
        )
        self.wait(2)
